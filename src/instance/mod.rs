mod debug_callback;
mod instance;
mod layers;

use std::{error::Error, ffi::CStr, sync::Arc};

use ash::vk;

use crate::{
    device::{physical_device::PhysicalDevice, queue::QueueFamilyInfo},
    errors::DeviceError,
    render::surface::{Surface, window::WindowSystemRawHandlers},
};

pub(crate) struct Instance {
    pub handle: ash::Instance,
    pub entry: ash::Entry,
    pub ws_handlers: Option<WindowSystemRawHandlers>,
    _debug_utils_messanger: Option<debug_callback::DebugUtilsMessanger>,
}

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
        match unsafe { self.handle.enumerate_physical_devices() } {
            Ok(devices) => Ok(devices),
            Err(e) => Err(Box::new(DeviceError::new(format!(
                "cannot enumerate physical devices: {e}"
            )))),
        }
    }

    pub fn new(ws_handlers: Option<WindowSystemRawHandlers>) -> Result<Arc<Self>, Box<dyn Error>> {
        let entry = match unsafe { ash::Entry::load() } {
            Ok(entry) => entry,
            Err(e) => return Err(Box::new(DeviceError::new(format!("{e}")))),
        };

        let instance = instance::new_in(&entry, ws_handlers)?;

        let _debug_utils_messanger = {
            #[cfg(debug_assertions)]
            match debug_callback::create_debug_utils_messanger(&entry, &instance) {
                Ok(debug_utils_messanger) => {
                    dbg!("debug_utils_messanger created");
                    Some(debug_utils_messanger)
                }
                Err(e) => {
                    dbg!("debug_utils_messanger creation error: {e}");
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
