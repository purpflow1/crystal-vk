pub mod physical_device;
pub mod queue;

use ash::vk;
use std::{error::Error, ffi::CStr, sync::Arc};

use crate::device::{
    physical_device::PhysicalDevice,
    queue::{Queue, QueuePool},
};

pub struct Device {
    pub(crate) handle: ash::Device,
    pub(crate) physical_device: PhysicalDevice,
    pub(crate) extensions: Vec<String>,
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
    /// Extension `VK_KHR_portability_subset` is enabled by default for apple target.
    /// Unsupported extensions are disabled automatically
    pub fn new(
        physical_device: PhysicalDevice,
        features: vk::PhysicalDeviceFeatures,
        extensions: Vec<&'static CStr>,
    ) -> Result<(Arc<Self>, QueuePool), Box<dyn Error>> {
        let (device_handler, extensions) = physical_device.create_device(features, extensions)?;

        let device = Arc::new(Self {
            handle: device_handler,
            physical_device,
            extensions,
        });

        let queues = Queue::instantiate(device.clone());
        Ok((device, queues))
    }

    /// # Safety
    /// Use at your own risk
    pub unsafe fn as_raw(&self) -> &ash::Device {
        &self.handle
    }
}
