mod debug_callback;
mod layers;

use std::{error::Error, sync::Arc};

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};

use crate::{
    device::physical_device::PhysicalDevice,
    render::surface::{self, Surface},
};

pub struct Instance {
    pub(crate) handle: ash::Instance,
    pub(crate) entry: ash::Entry,
    _debug_utils_messanger: Option<debug_callback::DebugUtilsMessanger>,
    window: Option<(RawWindowHandle, RawDisplayHandle)>,
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
    pub fn create_surface(self: &Arc<Self>) -> Result<Arc<surface::Surface>, Box<dyn Error>> {
        surface::Surface::new(self.clone(), self.window.expect(""))
    }

    pub fn enumerate_physical_devices(
        self: &Arc<Self>,
        surface: Option<Arc<Surface>>,
    ) -> Result<Vec<PhysicalDevice>, Box<dyn Error>> {
        let physical_devices_raw = unsafe { self.handle.enumerate_physical_devices() }?;

        let physical_devices =
            unsafe { PhysicalDevice::new(self.clone(), surface.clone(), physical_devices_raw) }?;

        Ok(physical_devices)
    }

    pub fn new_entry(
        display: Option<(RawWindowHandle, RawDisplayHandle)>,
        entry: ash::Entry,
    ) -> Result<Arc<Self>, Box<dyn Error>> {
        let mut instance_extensions = vec![
            #[cfg(debug_assertions)]
            vk::EXT_DEBUG_UTILS_NAME.as_ptr(),
            #[cfg(target_vendor = "apple")]
            vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr(),
        ];

        let display_extensions;

        if let Some(display) = display {
            display_extensions = ash_window::enumerate_required_extensions(display.1)?;

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
                Ok(debug_utils_messanger) => Some(debug_utils_messanger),
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
            _debug_utils_messanger,
            window: display,
        }))
    }

    fn new_in(
        display: Option<(RawWindowHandle, RawDisplayHandle)>,
    ) -> Result<Arc<Self>, Box<dyn Error>> {
        let entry = unsafe { ash::Entry::load() }?;
        Self::new_entry(display, entry)
    }

    pub fn new() -> Result<Arc<Self>, Box<dyn Error>> {
        Self::new_in(None)
    }

    pub fn new_window<T: HasWindowHandle + HasDisplayHandle>(
        window: &T,
    ) -> Result<Arc<Self>, Box<dyn Error>> {
        Self::new_in(Some((
            window.window_handle().unwrap().as_raw(),
            window.display_handle().unwrap().as_raw(),
        )))
    }
}
