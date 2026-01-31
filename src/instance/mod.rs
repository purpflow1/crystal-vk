mod debug_callback;
mod inner;
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

        let instance = inner::new_in(&entry, ws_handlers)?;

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
