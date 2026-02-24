pub mod physical_device;
pub mod queue;

use ash::vk;
use std::{error::Error, sync::Arc};

use crate::{
    device::{
        physical_device::PhysicalDevice,
        queue::{Queue, QueuePool},
    },
    instance,
    render::surface,
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
    pub fn new(
        physical_device: Arc<PhysicalDevice>,
        features: vk::PhysicalDeviceFeatures,
    ) -> Result<(Arc<Self>, QueuePool), Box<dyn Error>> {
        let (device_handler, extensions) =
            physical_device.create_device(features, physical_device.surface.is_some())?;

        let device = Arc::new(Self {
            handle: device_handler,
            instance: physical_device.instance.clone(),
            surface: physical_device.surface.clone(),
            physical_device,
            extensions,
        });

        let queues = Queue::instantiate(device.clone());
        Ok((device, queues))
    }
}
