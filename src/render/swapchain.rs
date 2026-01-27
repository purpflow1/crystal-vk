use std::{
    error::Error,
    sync::{Arc, Mutex, RwLock},
};

use ash::vk;

use crate::{
    device::{Device, queue::Queue},
    error,
    errors::{DeviceError, SwapChainError},
    image::{Image, ImageInfo},
    sync::SwapchainFuture,
};

struct SwapchainInfo {
    // Graphics pipeline requires this format of indices
    queue_family_indices: [u32; 2],
    extent: vk::Extent2D,
    present_mode: vk::PresentModeKHR,
    vsync: bool,

    pub image_count: u32,
    pub surface_format: vk::SurfaceFormatKHR,
}

impl SwapchainInfo {
    fn new(device: Arc<Device>, vsync: bool) -> Result<Arc<Self>, Box<dyn Error>> {
        let queue_family_indices = [
            match device
                .physical_device
                .queue_families_info
                .iter()
                .find(|info| info.flags.intersects(vk::QueueFlags::GRAPHICS))
            {
                Some(info) => info.index,
                None => {
                    return error!(DeviceError, "device does not support graphics queue");
                }
            },
            match device
                .physical_device
                .queue_families_info
                .iter()
                .find(|info| info.present_support)
            {
                Some(info) => info.index,
                None => {
                    return error!(DeviceError, "device does not support presentation");
                }
            },
        ];

        let swap_chain_support_details =
            match device.physical_device.swap_chain_support_details.clone() {
                Some(s) => s,
                None => return error!(SwapChainError, "surface haven't passed into device"),
            };

        let swap_surface_format = match swap_chain_support_details.formats.iter().find(|format| {
            format.format == vk::Format::B8G8R8A8_UNORM
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        }) {
            Some(&format) => format,
            None => {
                return error!(SwapChainError, "not found required swap surface format");
            }
        };

        let swap_extent =
            if swap_chain_support_details.capabilities.current_extent.width != u32::MAX {
                swap_chain_support_details.capabilities.current_extent
            } else {
                todo!("unknown surface extent, framebuffer extent is required");
            };

        let image_count = {
            let max_image_count = swap_chain_support_details.capabilities.max_image_count;
            let min_image_count = swap_chain_support_details.capabilities.min_image_count;
            if max_image_count > 0 {
                max_image_count
            } else {
                min_image_count
            }
        };

        let swap_present_mode = match swap_chain_support_details
            .present_modes
            .iter()
            .find(|&&mode| mode == vk::PresentModeKHR::MAILBOX)
        {
            Some(&mode) => mode,
            None => {
                if vsync {
                    vk::PresentModeKHR::FIFO
                } else {
                    vk::PresentModeKHR::IMMEDIATE
                }
            }
        };

        Ok(Arc::new(Self {
            queue_family_indices,
            vsync,
            surface_format: swap_surface_format,
            extent: swap_extent,
            present_mode: swap_present_mode,
            image_count,
        }))
    }
}

pub struct Swapchain {
    pub(crate) swapchain_info: Arc<SwapchainInfo>,
    pub(crate) swapchain: ash::khr::swapchain::Device,
    pub(crate) swapchain_khr: vk::SwapchainKHR,
    pub image_sequence: Vec<Arc<Image>>,

    pub present_queue: Arc<Mutex<Queue>>,
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            let mut lock = self.present_queue.lock().unwrap();
            lock.wait_idle().unwrap();
            self.swapchain.destroy_swapchain(self.swapchain_khr, None);
        }
    }
}

impl Swapchain {
    pub fn from_old(old: Arc<Self>) -> Result<Arc<Self>, Box<dyn Error>> {
        let device = old.present_queue.lock().unwrap().device.clone();
        let swapchain_info = SwapchainInfo::new(device, old.swapchain_info.vsync)?;
        Self::new_in(old.present_queue.clone(), swapchain_info, old.swapchain_khr)
    }

    pub fn new(present_queue: Arc<Mutex<Queue>>, vsync: bool) -> Result<Arc<Self>, Box<dyn Error>> {
        let device = present_queue.lock().unwrap().device.clone();
        let swapchain_info = SwapchainInfo::new(device, vsync)?;
        Self::new_in(present_queue, swapchain_info, vk::SwapchainKHR::null())
    }

    fn new_in(
        present_queue: Arc<Mutex<Queue>>,
        swapchain_info: Arc<SwapchainInfo>,
        old_swapchain: vk::SwapchainKHR,
    ) -> Result<Arc<Self>, Box<dyn Error>> {
        let device = present_queue.lock().unwrap().device.clone();

        let swapchain = ash::khr::swapchain::Device::new(&device.instance.handle, &device.handle);

        let mut compression_control = vk::ImageCompressionControlEXT::default()
            .flags(vk::ImageCompressionFlagsEXT::FIXED_RATE_DEFAULT);

        let support_details = device
            .physical_device
            .swap_chain_support_details
            .clone()
            .unwrap();

        let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(device.surface.surface_khr)
            .min_image_count(swapchain_info.image_count)
            .image_format(swapchain_info.surface_format.format)
            .image_color_space(swapchain_info.surface_format.color_space)
            .image_extent(swapchain_info.extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .pre_transform(support_details.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .present_mode(swapchain_info.present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);

        if swapchain_info.queue_family_indices[0] != swapchain_info.queue_family_indices[1] {
            swapchain_create_info = swapchain_create_info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&swapchain_info.queue_family_indices)
        }

        if device.extensions.iter().any(|ext| {
            ext.as_str()
                == vk::EXT_IMAGE_COMPRESSION_CONTROL_SWAPCHAIN_NAME
                    .to_str()
                    .unwrap()
        }) {
            swapchain_create_info = swapchain_create_info.push_next(&mut compression_control);
        }

        let swapchain_khr =
            match unsafe { swapchain.create_swapchain(&swapchain_create_info, None) } {
                Ok(swapchain_khr) => swapchain_khr,
                Err(e) => {
                    return error!(SwapChainError, "cannot create swapchain: {e}");
                }
            };

        let swapchain_images = match unsafe { swapchain.get_swapchain_images(swapchain_khr) } {
            Ok(images) => images,
            Err(e) => {
                return error!(SwapChainError, "cannot get swapchain images: {e}");
            }
        };

        let mut images = vec![];

        for &swapchain_image in &swapchain_images {
            let create_info = vk::ImageViewCreateInfo::default()
                .image(swapchain_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(swapchain_info.surface_format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            let image_view = match unsafe { device.handle.create_image_view(&create_info, None) } {
                Ok(image_view) => image_view,
                Err(e) => {
                    return error!(DeviceError, "cannot create image view: {e}");
                }
            };

            images.push(Image::new_swapchain(
                device.clone(),
                image_view,
                [swapchain_info.extent.width, swapchain_info.extent.height],
                swapchain_info.surface_format.format,
            ));
        }

        Ok(Arc::new(Swapchain {
            swapchain_info,
            swapchain,
            swapchain_khr,
            image_sequence: images,
            present_queue,
        }))
    }
}
