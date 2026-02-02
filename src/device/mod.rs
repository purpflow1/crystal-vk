pub mod physical_device;
pub mod queue;

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{error::Error, sync::Arc};

use crate::{
    device::{
        physical_device::PhysicalDevice,
        queue::{Queue, QueuePool},
    },
    instance,
    render::surface::{
        self,
        window::{self, WindowSystemRawHandlers},
    },
};

pub struct Device {
    pub(crate) handle: ash::Device,
    pub(crate) instance: Arc<instance::Instance>,
    pub(crate) physical_device: Arc<physical_device::PhysicalDevice>,
    pub(crate) extensions: Vec<String>,
    pub(crate) surface: Option<Arc<surface::Surface>>,
}

unsafe impl Send for Device {}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.handle.device_wait_idle().unwrap();
            self.handle.destroy_device(None)
        }
    }
}

impl Device {
    pub fn compute<P>(
        physical_device_pick_predicate: P,
    ) -> Result<(Arc<Self>, QueuePool), Box<dyn Error>>
    where
        P: Fn(Vec<Arc<PhysicalDevice>>) -> Arc<PhysicalDevice>,
    {
        let device = Self::new_in(physical_device_pick_predicate, None)?;
        let queues = Queue::instantiate(device.clone());
        Ok((device, queues))
    }

    pub fn with_present<T: HasWindowHandle + HasDisplayHandle, P>(
        physical_device_pick_predicate: P,
        window: &T,
    ) -> Result<(Arc<Self>, QueuePool), Box<dyn Error>>
    where
        P: Fn(Vec<Arc<PhysicalDevice>>) -> Arc<PhysicalDevice>,
    {
        let handlers = WindowSystemRawHandlers::new(window)?;
        let device = Self::new_in(physical_device_pick_predicate, Some(handlers))?;
        let queues = Queue::instantiate(device.clone());
        Ok((device, queues))
    }

    fn new_in<P>(
        physical_device_pick_predicate: P,
        handlers: Option<window::WindowSystemRawHandlers>,
    ) -> Result<Arc<Self>, Box<dyn Error>>
    where
        P: Fn(Vec<Arc<PhysicalDevice>>) -> Arc<PhysicalDevice>,
    {
        let instance = instance::Instance::new(handlers)?;

        let surface = match handlers {
            Some(_) => Some(surface::Surface::new(instance.clone())?),
            None => None,
        };

        let physical_devices_raw = unsafe { instance.enumerate_physical_devices() }?;
        let physical_devices = unsafe {
            PhysicalDevice::new(instance.clone(), surface.clone(), physical_devices_raw)
        }?;

        let physical_device = physical_device_pick_predicate(physical_devices);

        let (device_handler, extensions) = physical_device.create_device(
            vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true),
            handlers.is_some(),
        )?;

        let device = Arc::new(Self {
            handle: device_handler,
            instance,
            surface,
            physical_device,
            extensions,
        });

        Ok(device)
    }
}
