mod debug_callback;
mod layers;

use std::{error::Error, sync::Arc};

use ash::vk;

use crate::render::surface::window::WindowSystemRawHandlers;

pub(crate) struct Instance {
    pub handle: ash::Instance,
    pub entry: ash::Entry,
    pub ws_handlers: Option<WindowSystemRawHandlers>,
    _debug_utils_messanger: Option<debug_callback::DebugUtilsMessanger>,
}

unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

impl std::fmt::Debug for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("")
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self._debug_utils_messanger = None;
            self.handle.destroy_instance(None);
        }
    }
}

impl Instance {
    pub unsafe fn enumerate_physical_devices(
        &self,
    ) -> Result<Vec<vk::PhysicalDevice>, Box<dyn Error>> {
        Ok(unsafe { self.handle.enumerate_physical_devices() }?)
    }

    pub fn new(ws_handlers: Option<WindowSystemRawHandlers>) -> Result<Arc<Self>, Box<dyn Error>> {
        let entry = unsafe { ash::Entry::load() }?;

        let mut instance_extensions = vec![
            #[cfg(debug_assertions)]
            vk::EXT_DEBUG_UTILS_NAME.as_ptr(),
            #[cfg(target_vendor = "apple")]
            vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr(),
        ];

        let display_extensions;

        if let Some(ws_handlers) = ws_handlers {
            display_extensions = ash_window::enumerate_required_extensions(ws_handlers.display)?;

            display_extensions
                .iter()
                .for_each(|ext| instance_extensions.push(*ext));
        }

        let flags = if cfg!(target_vendor = "apple") {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };

        pub(super) const API_VERSION_LATEST: u32 = u32::MAX;

        let app_info = vk::ApplicationInfo::default().api_version(API_VERSION_LATEST);
        #[cfg(not(debug_assertions))]
        let create_info = vk::InstanceCreateInfo::default()
            .flags(flags)
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions);

        #[cfg(debug_assertions)]
        let layers;
        #[cfg(debug_assertions)]
        let layers_pp: Vec<*const i8>;

        #[cfg(debug_assertions)]
        let create_info = {
            layers = layers::get_supported_validation_layers(&entry);

            if layers.is_empty() {
                println!(
                    "No validation layers found!
                Vulkan SDK should be installed for proper debug.
                Visit https://vulkan.lunarg.com/"
                );

                vk::InstanceCreateInfo::default()
                    .flags(flags)
                    .application_info(&app_info)
                    .enabled_extension_names(&instance_extensions)
            } else {
                layers_pp = layers.iter().map(|x| x.as_ptr() as *const _).collect();

                let mut create_info = vk::InstanceCreateInfo::default()
                    .flags(flags)
                    .application_info(&app_info)
                    .enabled_extension_names(&instance_extensions);

                create_info.pp_enabled_layer_names = layers_pp.as_ptr() as *const _;
                create_info.enabled_layer_count = layers_pp.len() as u32;

                create_info
            }
        };

        let instance = unsafe { entry.create_instance(&create_info, None) }?;

        let _debug_utils_messanger = {
            #[cfg(debug_assertions)]
            match debug_callback::create_debug_utils_messanger(&entry, &instance) {
                Ok(debug_utils_messanger) => {
                    dbg!("debug_utils_messanger created");
                    Some(debug_utils_messanger)
                }
                Err(e) => {
                    dbg!(e);
                    None
                }
            }

            #[cfg(not(debug_assertions))]
            None
        };

        Ok(Arc::new(Self {
            handle: instance,
            entry,
            ws_handlers,
            _debug_utils_messanger,
        }))
    }
}
