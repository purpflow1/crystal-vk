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
    pub rt_props: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
    pub features: vk::PhysicalDeviceFeatures,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_family_properties: Vec<vk::QueueFamilyProperties>,
    pub queue_families_info: Vec<QueueFamilyInfo>,
}

pub struct PhysicalDevice {
    pub handle: vk::PhysicalDevice,
    pub info: PhysicalDeviceInfo,
    pub swap_chain_support_details: Option<SwapChainSupportDetails>,
    pub(crate) instance: Arc<Instance>,
    pub(crate) surface: Option<Arc<Surface>>,
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

    /// # Safety
    /// - `instance` must be a valid Vulkan instance handle.
    /// - `physical_devices` must be valid handles retrieved from the instance.
    /// - If `surface` is provided, it must be a valid surface created from the same instance.
    pub(crate) unsafe fn new(
        instance: Arc<Instance>,
        surface: Option<Arc<Surface>>,
        physical_devices: Vec<vk::PhysicalDevice>,
    ) -> Result<Vec<Arc<Self>>, Box<dyn Error>> {
        let mut devices = Vec::with_capacity(physical_devices.len());

        for &device_handle in &physical_devices {
            // Gather basic device properties
            let mut rt_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
            let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut rt_props);
            let (properties, features, memory_properties, queue_family_properties) = unsafe {
                let features = instance.handle.get_physical_device_features(device_handle);
                let properties = instance
                    .handle
                    .get_physical_device_properties(device_handle);
                let memory_properties = instance
                    .handle
                    .get_physical_device_memory_properties(device_handle);
                let queue_family_properties = instance
                    .handle
                    .get_physical_device_queue_family_properties(device_handle);

                instance
                    .handle
                    .get_physical_device_properties2(device_handle, &mut properties2);
                (
                    properties,
                    features,
                    memory_properties,
                    queue_family_properties,
                )
            };

            // Device name (panics if not valid UTF-8, preserving original behavior)
            let device_name = unsafe {
                CStr::from_ptr(properties.device_name.as_ptr())
                    .to_str()
                    .expect("Device name is not valid UTF-8")
                    .to_string()
            };

            // Query swap chain support if a surface is available (panics on error, preserving original behavior)
            let swap_chain_support_details = surface.as_ref().map(|s| {
                Self::query_swap_chain_support(device_handle, s.clone())
                    .expect("Failed to query swap chain support")
            });

            // Build queue family info (preserves the original logic, including the index bug)
            let queue_families_info = Self::build_queue_families_info(
                &queue_family_properties,
                device_handle,
                surface.as_ref(),
            );

            let info = PhysicalDeviceInfo {
                name: device_name,
                properties,
                rt_props,
                features,
                memory_properties,
                queue_family_properties,
                queue_families_info,
            };

            devices.push(Arc::new(Self {
                handle: device_handle,
                info,
                swap_chain_support_details,
                instance: instance.clone(),
                surface: surface.clone(),
            }));
        }

        Ok(devices)
    }

    /// Builds the queue family information for a physical device.
    /// This function replicates the original (buggy) logic exactly to preserve external behavior.
    /// The index used is the enumeration index after filtering, not the real queue family index.
    fn build_queue_families_info(
        queue_family_properties: &[vk::QueueFamilyProperties],
        device_handle: vk::PhysicalDevice,
        surface: Option<&Arc<Surface>>,
    ) -> Vec<QueueFamilyInfo> {
        let mut queue_families = Vec::<QueueFamilyInfo>::new();

        for (idx, family) in queue_family_properties
            .iter()
            .enumerate()
            .filter(|(_, f)| f.queue_count > 0)
        {
            let index = idx as u32; // ! This is the enumeration index, not the real queue family index!

            let present_support = surface.is_some_and(|surface| unsafe {
                surface
                    .surface
                    .get_physical_device_surface_support(device_handle, index, surface.surface_khr)
                    .expect("Failed to query surface support")
            });

            let mut to_push = false;

            // Update existing entries or decide to push a new one
            for info in &mut queue_families {
                if info.index == index {
                    info.flags |= family.queue_flags;
                    info.present_support = present_support;
                } else if !info.flags.intersects(family.queue_flags) {
                    to_push = true;
                }
            }

            if to_push || queue_families.is_empty() {
                queue_families.push(QueueFamilyInfo {
                    flags: family.queue_flags,
                    index,
                    queue_count: family.queue_count,
                    present_support,
                });
            }
        }

        queue_families
    }

    fn query_extensions_support(
        &self,
        extensions: Vec<&'static CStr>,
    ) -> Result<Vec<&CStr>, Box<dyn Error>> {
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
        extensions: Vec<&'static CStr>,
    ) -> Result<(ash::Device, Vec<String>), Box<dyn Error>> {
        let extensions = self.query_extensions_support(extensions)?;

        let mut device_queue_create_infos = Vec::with_capacity(self.info.queue_families_info.len());

        let queue_priorities = self
            .info
            .queue_families_info
            .iter()
            .map(|info| vec![1.; info.queue_count as usize])
            .collect::<Vec<_>>();

        for (idx, info) in self.info.queue_families_info.iter().enumerate() {
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

        let mut accel = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
            .acceleration_structure(true);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&device_queue_create_infos)
            .enabled_features(&features)
            .enabled_extension_names(&extension_names)
            .push_next(&mut accel);

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
