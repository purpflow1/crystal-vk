use std::{error::Error, ffi::CStr, sync::Arc};

use ash::vk;

use crate::{device::queue::QueueFamilyInfo, instance::Instance, render::surface::Surface};

#[derive(Clone, Debug)]
pub struct SwapChainSupportDetails {
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
    pub capabilities: vk::SurfaceCapabilitiesKHR,
}

pub struct PhysicalDeviceInfo {
    pub name: String,
    pub properties: vk::PhysicalDeviceProperties,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_family_properties: Vec<vk::QueueFamilyProperties>,
    pub queue_families_info: Vec<QueueFamilyInfo>,
}

pub struct PhysicalDevice {
    pub handle: vk::PhysicalDevice,
    pub info: PhysicalDeviceInfo,
    pub swap_chain_support_details: Option<SwapChainSupportDetails>,
    pub(crate) instance: Arc<Instance>,
}

impl std::fmt::Debug for PhysicalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("PhysicalDevice: {}", self.info.name).as_str())
    }
}

impl PhysicalDevice {
    pub(crate) fn find_memory_type_index(
        &self,
        flags: vk::MemoryPropertyFlags,
        type_filter: u32,
    ) -> Result<u32, Box<dyn Error>> {
        for i in 0..self.info.memory_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && (self.info.memory_properties.memory_types[i as usize].property_flags & flags)
                    == flags
            {
                return Ok(i);
            }
        }

        Err("cannot find suitable memory type".into())
    }

    pub(crate) fn query_swap_chain_support(
        device: vk::PhysicalDevice,
        surface: Arc<Surface>,
    ) -> Result<SwapChainSupportDetails, Box<dyn Error>> {
        let formats = unsafe {
            surface
                .surface
                .get_physical_device_surface_formats(device, surface.surface_khr)
        }?;

        let capabilities = unsafe {
            surface
                .surface
                .get_physical_device_surface_capabilities(device, surface.surface_khr)
        }?;

        let present_modes = unsafe {
            surface
                .surface
                .get_physical_device_surface_present_modes(device, surface.surface_khr)
        }?;

        Ok(SwapChainSupportDetails {
            formats,
            present_modes,
            capabilities,
        })
    }

    pub(crate) unsafe fn new(
        instance: Arc<Instance>,
        surface: Option<Arc<Surface>>,
        physical_devices: Vec<vk::PhysicalDevice>,
    ) -> Result<Vec<Arc<Self>>, Box<dyn Error>> {
        let physical_devices = physical_devices
            .iter()
            .map(|physical_device| {
                let (
                    properties,
                    device_name,
                    memory_properties,
                    queue_family_properties,
                    swap_chain_support_details,
                ) = unsafe {
                    let properties = instance
                        .handle
                        .get_physical_device_properties(*physical_device);
                    (
                        properties,
                        std::ffi::CStr::from_ptr(properties.device_name.as_ptr())
                            .to_str()
                            .unwrap(),
                        instance
                            .handle
                            .get_physical_device_memory_properties(*physical_device),
                        instance
                            .handle
                            .get_physical_device_queue_family_properties(*physical_device),
                        surface.clone().map(|surface| {
                            Self::query_swap_chain_support(*physical_device, surface).unwrap()
                        }),
                    )
                };

                let mut queue_families = Vec::<QueueFamilyInfo>::new();

                for (index, family) in queue_family_properties
                    .iter()
                    .filter(|f| f.queue_count > 0)
                    .enumerate()
                {
                    let index = index as u32;

                    let present_support = unsafe {
                        match surface.clone() {
                            Some(surface) => surface
                                .surface
                                .get_physical_device_surface_support(
                                    *physical_device,
                                    index,
                                    surface.surface_khr,
                                )
                                .unwrap(),
                            None => false,
                        }
                    };

                    let mut to_push = false;

                    queue_families.iter_mut().for_each(|info| {
                        if info.index == index {
                            info.flags |= family.queue_flags;
                            info.present_support = present_support
                        } else if !info.flags.intersects(family.queue_flags) {
                            to_push = true;
                        }
                    });

                    if to_push || queue_families.is_empty() {
                        queue_families.push(QueueFamilyInfo {
                            flags: family.queue_flags,
                            index,
                            queue_count: family.queue_count,
                            present_support,
                        });
                    }
                }

                Arc::new(Self {
                    handle: *physical_device,
                    info: PhysicalDeviceInfo {
                        name: device_name.to_string(),
                        properties,
                        memory_properties,
                        queue_family_properties,
                        queue_families_info: queue_families,
                    },

                    swap_chain_support_details,
                    instance: instance.clone(),
                })
            })
            .collect();

        Ok(physical_devices)
    }

    fn query_extensions_support(
        &self,
        enable_swapchain: bool,
    ) -> Result<Vec<&CStr>, Box<dyn Error>> {
        let extensions = if enable_swapchain {
            vec![
                vk::KHR_SWAPCHAIN_NAME,
                vk::KHR_ACCELERATION_STRUCTURE_NAME,
                vk::KHR_RAY_TRACING_PIPELINE_NAME,
                vk::KHR_DEFERRED_HOST_OPERATIONS_NAME,
                vk::EXT_IMAGE_COMPRESSION_CONTROL_NAME,
                vk::EXT_IMAGE_COMPRESSION_CONTROL_SWAPCHAIN_NAME,
            ]
        } else {
            vec![vk::KHR_DEFERRED_HOST_OPERATIONS_NAME]
        };

        let mut supported_extensions = Vec::with_capacity(extensions.len());

        let extension_props = unsafe {
            self.instance
                .handle
                .enumerate_device_extension_properties(self.handle)
        }?;

        let device_supported_extensions: Vec<&std::ffi::CStr> = extension_props
            .iter()
            .map(|ext| ext.extension_name_as_c_str().unwrap())
            .collect();

        for sup_ext in &device_supported_extensions {
            if let Some(ext) = extensions.iter().find(|&req_ext| *req_ext == *sup_ext) {
                supported_extensions.push(*ext)
            }
        }

        Ok(supported_extensions)
    }

    pub(crate) fn create_device(
        &self,
        features: vk::PhysicalDeviceFeatures,
        enable_swapchain: bool,
    ) -> Result<(ash::Device, Vec<String>), Box<dyn Error>> {
        let extensions = self.query_extensions_support(enable_swapchain)?;

        dbg!(self.info.queue_families_info.clone());

        let mut device_queue_create_infos = Vec::with_capacity(self.info.queue_families_info.len());

        let queue_priorities = self
            .info
            .queue_families_info
            .iter()
            .map(|info| vec![1.; info.queue_count as usize])
            .collect::<Vec<_>>();

        for (idx, info) in self.info.queue_families_info.iter().enumerate() {
            // TODO add more queues
            device_queue_create_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(info.index)
                    .queue_priorities(&queue_priorities[idx]),
            )
        }

        let portability = if cfg!(target_os = "macos") {
            vec![vk::KHR_PORTABILITY_SUBSET_NAME.as_ptr()]
        } else {
            Vec::with_capacity(0)
        };

        let extension_names: Vec<_> = extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .chain(portability)
            .collect();

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&device_queue_create_infos)
            .enabled_features(&features)
            .enabled_extension_names(&extension_names);

        Ok((
            unsafe {
                self.instance
                    .handle
                    .create_device(self.handle, &device_create_info, None)
            }?,
            extensions
                .iter()
                .map(|ext| ext.to_str().unwrap().to_string())
                .collect(),
        ))
    }

    pub(crate) fn find_depth_format(
        &self,
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> Option<vk::Format> {
        let mut depth_format = None;

        for format in [
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ] {
            let properties = unsafe {
                self.instance
                    .handle
                    .get_physical_device_format_properties(self.handle, format)
            };

            if (tiling == vk::ImageTiling::LINEAR
                && (properties.linear_tiling_features & features) == features)
                || tiling == vk::ImageTiling::OPTIMAL
                    && (properties.optimal_tiling_features & features) == features
            {
                depth_format = Some(format);
                break;
            }
        }

        depth_format
    }
}
