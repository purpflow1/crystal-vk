use std::{cell::Cell, error::Error, sync::Arc};

use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::render::swapchain::Swapchain;

pub struct Surface {
    pub(crate) surface: ash::khr::surface::Instance,
    pub(crate) surface_khr: ash::vk::SurfaceKHR,
    // lifetime depends on Surface
    pub(crate) swapchain: Cell<Option<Arc<Swapchain>>>,
    _instance: Arc<crate::instance::Instance>,
}

unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { self.surface.destroy_surface(self.surface_khr, None) }
    }
}

impl Surface {
    pub fn new(
        instance: Arc<crate::instance::Instance>,
        window: (RawWindowHandle, RawDisplayHandle),
    ) -> Result<Arc<Self>, Box<dyn Error>> {
        let surface = ash::khr::surface::Instance::new(&instance.entry, &instance.handle);
        let surface_khr = unsafe {
            ash_window::create_surface(&instance.entry, &instance.handle, window.1, window.0, None)
                .unwrap()
        };

        Ok(Arc::new(Self {
            _instance: instance,
            swapchain: Cell::new(None),
            surface,
            surface_khr,
        }))
    }
}
