use std::{error::Error, sync::Arc};

use ash::vk;

use crate::{device::Device, error, errors::DeviceError};

pub enum ImageType {
    Swapchain,
    Texture,
    FramebufferColor,
    FramebufferDepth,
}

pub struct ImageInfo {
    pub(crate) extent: [u32; 2],
    pub(crate) typ: ImageType,
    pub(crate) format: vk::Format,
}

struct ImageCreateInfo {
    pub width: u32,
    pub height: u32,
    pub anisotropy_texels: f32,
    pub generate_mips: bool,
    pub image_type: ImageType,

    pub samples: vk::SampleCountFlags,
    pub format: vk::Format,
    pub tiling: vk::ImageTiling,
    pub aspect_mask: vk::ImageAspectFlags,
    pub usage: vk::ImageUsageFlags,
    pub mem_property: vk::MemoryPropertyFlags,
}

pub struct Image {
    pub(crate) image_view: vk::ImageView,
    pub(crate) handle: vk::Image,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) info: ImageInfo,

    pub device: Arc<Device>,
}

impl Drop for Image {
    fn drop(&mut self) {
        if self.memory != vk::DeviceMemory::null() {
            unsafe { self.device.handle.free_memory(self.memory, None) }
        }
        if self.image_view != vk::ImageView::null() {
            unsafe { self.device.handle.destroy_image_view(self.image_view, None) }
        }
        if self.handle != vk::Image::null() {
            unsafe { self.device.handle.destroy_image(self.handle, None) }
        }
    }
}

impl Image {
    pub(crate) fn new_framebuffer_color(
        device: Arc<Device>,
        extent: [u32; 2],
        image_format: vk::Format,
        samples: vk::SampleCountFlags,
    ) -> Result<Arc<Self>, Box<dyn Error>> {
        let image_create_info = ImageCreateInfo {
            width: extent[0],
            height: extent[1],
            generate_mips: false,
            anisotropy_texels: 1.,
            format: image_format,
            samples,
            tiling: vk::ImageTiling::OPTIMAL,
            aspect_mask: vk::ImageAspectFlags::COLOR,
            usage: vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            mem_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            image_type: ImageType::FramebufferColor,
        };

        Self::new_in(device, image_create_info)
    }

    pub(crate) fn new_framebuffer_depth(
        device: Arc<Device>,
        extent: [u32; 2],
        samples: vk::SampleCountFlags,
    ) -> Result<Arc<Self>, Box<dyn Error>> {
        let tiling = vk::ImageTiling::OPTIMAL;
        let features = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT;

        let depth_format = match device.physical_device.find_depth_format(tiling, features) {
            Some(format) => format,
            None => {
                return error!(DeviceError, "cannot find supported depth format");
            }
        };

        let image_create_info = ImageCreateInfo {
            width: extent[0],
            height: extent[1],
            generate_mips: false,
            anisotropy_texels: 1.,
            format: depth_format,
            samples,
            tiling,
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            mem_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            image_type: ImageType::FramebufferDepth,
        };

        Self::new_in(device, image_create_info)
    }

    pub(crate) fn new_swapchain(
        device: Arc<Device>,
        image_view: vk::ImageView,
        extent: [u32; 2],
        image_format: vk::Format,
    ) -> Arc<Self> {
        Arc::new(Self {
            image_view,
            handle: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            info: ImageInfo {
                extent,
                typ: ImageType::Swapchain,
                format: image_format,
            },
            device,
        })
    }

    fn new_in(
        device: Arc<Device>,
        image_create_info: ImageCreateInfo,
    ) -> Result<Arc<Self>, Box<dyn Error>> {
        let extent = vk::Extent3D::default()
            .width(image_create_info.width)
            .height(image_create_info.height)
            .depth(1);

        let mip_levels = if image_create_info.generate_mips {
            (image_create_info.height as f32)
                .max(image_create_info.width as f32)
                .log2()
                .floor() as u32
        } else {
            1
        };

        let ty = vk::ImageType::TYPE_2D;

        let mut compression_control = vk::ImageCompressionControlEXT::default()
            .flags(vk::ImageCompressionFlagsEXT::FIXED_RATE_DEFAULT);

        let create_info = vk::ImageCreateInfo::default()
            .image_type(ty)
            .extent(extent)
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(image_create_info.format)
            .tiling(image_create_info.tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(image_create_info.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(image_create_info.samples)
            .push_next(&mut compression_control);

        let image = match unsafe { device.handle.create_image(&create_info, None) } {
            Ok(image) => image,
            Err(e) => {
                return error!(DeviceError, "cannot create image: {e}");
            }
        };

        let memory_requirements = unsafe { device.handle.get_image_memory_requirements(image) };

        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(device.physical_device.find_memory_type_index(
                image_create_info.mem_property,
                memory_requirements.memory_type_bits,
            )?);

        let image_memory =
            match unsafe { device.handle.allocate_memory(&memory_allocate_info, None) } {
                Ok(mem) => mem,
                Err(e) => {
                    return error!(DeviceError, "cannot allocate image memory: {e:?}");
                }
            };

        match unsafe { device.handle.bind_image_memory(image, image_memory, 0) } {
            Ok(_) => (),
            Err(e) => {
                return error!(DeviceError, "cannot bind image memory: {e}");
            }
        };

        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(image_create_info.format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(image_create_info.aspect_mask)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let image_view = match unsafe {
            device
                .handle
                .create_image_view(&image_view_create_info, None)
        } {
            Ok(image_view) => image_view,
            Err(e) => {
                return error!(DeviceError, "cannot create image view: {e}");
            }
        };

        Ok(Arc::new(Self {
            image_view,
            handle: image,
            memory: image_memory,
            info: ImageInfo {
                extent: [image_create_info.width, image_create_info.height],
                typ: image_create_info.image_type,
                format: image_create_info.format,
            },
            device,
        }))
    }
}
