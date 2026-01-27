pub mod window;

use std::{error::Error, sync::Arc};

use crate::errors::DeviceError;

pub(crate) struct Surface {
    pub surface: ash::khr::surface::Instance,
    pub surface_khr: ash::vk::SurfaceKHR,
    instance: Arc<crate::instance::Instance>,
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { self.surface.destroy_surface(self.surface_khr, None) }
    }
}

impl Surface {
    pub fn new(instance: Arc<crate::instance::Instance>) -> Result<Arc<Self>, Box<dyn Error>> {
        let ws_handlers = if let Some(h) = instance.ws_handlers {
            h
        } else {
            return Err(Box::new(DeviceError::new(format!(
                "cannot create surface: no window system handlers"
            ))));
        };

        let surface = ash::khr::surface::Instance::new(&instance.entry, &instance.handle);
        let surface_khr = unsafe {
            ash_window::create_surface(
                &instance.entry,
                &instance.handle,
                ws_handlers.display,
                ws_handlers.window,
                None,
            )
            .unwrap()
        };

        Ok(Arc::new(Self {
            instance,
            surface,
            surface_khr,
        }))
    }
}
