pub mod sampler;

use std::{cell::Cell, error::Error, sync::Arc};

use ash::vk;

use crate::device::Device;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum ImageType {
    Swapchain,
    Sampled,
    FramebufferColor,
    FramebufferDepth,
}

pub struct ImageInfo {
    pub extent: [u32; 2],
    pub typ: ImageType,
    pub format: vk::Format,
    pub layout: Cell<vk::ImageLayout>,
    pub mip_levels: u32,
    pub array_layers: u32,
}

pub struct ImageCreateInfo {
    pub width: u32,
    pub height: u32,
    pub generate_mips: bool,
    pub image_type: ImageType,

    pub samples: vk::SampleCountFlags,
    pub format: vk::Format,
    pub tiling: vk::ImageTiling,
    pub aspect_mask: vk::ImageAspectFlags,
    pub usage: vk::ImageUsageFlags,
    pub mem_property: vk::MemoryPropertyFlags,

    pub array_layers: u32,
    pub flags: vk::ImageCreateFlags,
    pub view_type: vk::ImageViewType,
}

impl Default for ImageCreateInfo {
    fn default() -> Self {
        Self {
            width: 1,
            height: 1,
            generate_mips: false,
            image_type: ImageType::Sampled,
            samples: vk::SampleCountFlags::TYPE_1,
            format: vk::Format::UNDEFINED,
            tiling: vk::ImageTiling::OPTIMAL,
            aspect_mask: vk::ImageAspectFlags::COLOR,
            usage: vk::ImageUsageFlags::empty(),
            mem_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            array_layers: 1,
            flags: vk::ImageCreateFlags::empty(),
            view_type: vk::ImageViewType::TYPE_2D,
        }
    }
}

pub struct Image {
    pub(crate) image_view: vk::ImageView,
    pub(crate) handle: vk::Image,
    pub(crate) memory: vk::DeviceMemory,
    pub info: ImageInfo,

    pub device: Arc<Device>,
}

unsafe impl Send for Image {}
unsafe impl Sync for Image {}

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
    /// Creates a simple 2D sampled image (existing behaviour).
    pub fn new(
        device: Arc<Device>,
        extent: [u32; 2],
        format: vk::Format,
    ) -> Result<Arc<Self>, Box<dyn Error>> {
        let format_properties = unsafe {
            device
                .instance
                .handle
                .get_physical_device_format_properties(device.physical_device.handle, format)
        };

        if format_properties.optimal_tiling_features
            & vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
            != vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
        {
            return Err(format!(
                "no suitable device for image linear filtering with format: {:?}",
                format
            )
            .into());
        }

        let create_info = ImageCreateInfo {
            width: extent[0],
            height: extent[1],
            generate_mips: false,
            image_type: ImageType::Sampled,
            format,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            aspect_mask: vk::ImageAspectFlags::COLOR,
            usage: vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            mem_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            array_layers: 1,
            flags: vk::ImageCreateFlags::empty(),
            view_type: vk::ImageViewType::TYPE_2D,
        };

        Self::new_in(device, create_info)
    }

    /// Creates a cubemap with 6 faces (array layers) and the CUBE_COMPATIBLE flag.
    pub fn new_cubemap(
        device: Arc<Device>,
        extent: [u32; 2],
        format: vk::Format,
        generate_mips: bool,
    ) -> Result<Arc<Self>, Box<dyn Error>> {
        let format_properties = unsafe {
            device
                .instance
                .handle
                .get_physical_device_format_properties(device.physical_device.handle, format)
        };

        if format_properties.optimal_tiling_features
            & vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
            != vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
        {
            return Err(format!(
                "no suitable device for image linear filtering with format: {:?}",
                format
            )
            .into());
        }

        let create_info = ImageCreateInfo {
            width: extent[0],
            height: extent[1],
            generate_mips,
            image_type: ImageType::Sampled,
            format,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            aspect_mask: vk::ImageAspectFlags::COLOR,
            usage: vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            mem_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            array_layers: 6,
            flags: vk::ImageCreateFlags::CUBE_COMPATIBLE,
            view_type: vk::ImageViewType::CUBE,
        };

        Self::new_in(device, create_info)
    }

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
            format: image_format,
            samples,
            tiling: vk::ImageTiling::OPTIMAL,
            aspect_mask: vk::ImageAspectFlags::COLOR,
            usage: vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            mem_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            image_type: ImageType::FramebufferColor,
            array_layers: 1,
            flags: vk::ImageCreateFlags::empty(),
            view_type: vk::ImageViewType::TYPE_2D,
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
                return Err("cannot find supported depth format".into());
            }
        };

        let image_create_info = ImageCreateInfo {
            width: extent[0],
            height: extent[1],
            generate_mips: false,
            format: depth_format,
            samples,
            tiling,
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            mem_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            image_type: ImageType::FramebufferDepth,
            array_layers: 1,
            flags: vk::ImageCreateFlags::empty(),
            view_type: vk::ImageViewType::TYPE_2D,
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
                layout: Cell::new(vk::ImageLayout::READ_ONLY_OPTIMAL),
                mip_levels: 0,
                array_layers: 1,
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
            .array_layers(image_create_info.array_layers)
            .format(image_create_info.format)
            .tiling(image_create_info.tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(image_create_info.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(image_create_info.samples)
            .flags(image_create_info.flags)
            .push_next(&mut compression_control);

        let image = unsafe { device.handle.create_image(&create_info, None) }?;

        let memory_requirements = unsafe { device.handle.get_image_memory_requirements(image) };

        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(device.physical_device.find_memory_type_index(
                image_create_info.mem_property,
                memory_requirements.memory_type_bits,
            )?);

        let image_memory = unsafe { device.handle.allocate_memory(&memory_allocate_info, None) }?;

        unsafe { device.handle.bind_image_memory(image, image_memory, 0) }?;

        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(image_create_info.view_type)
            .format(image_create_info.format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(image_create_info.aspect_mask)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(image_create_info.array_layers),
            );

        let image_view = unsafe {
            device
                .handle
                .create_image_view(&image_view_create_info, None)
        }?;

        Ok(Arc::new(Self {
            image_view,
            handle: image,
            memory: image_memory,
            info: ImageInfo {
                extent: [image_create_info.width, image_create_info.height],
                typ: image_create_info.image_type,
                format: image_create_info.format,
                layout: Cell::new(vk::ImageLayout::UNDEFINED),
                mip_levels,
                array_layers: image_create_info.array_layers,
            },
            device,
        }))
    }
}
