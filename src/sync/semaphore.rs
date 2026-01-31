use std::{error::Error, sync::Arc};

use ash::vk;

use crate::device::Device;

pub struct Semaphore {
    pub handle: vk::Semaphore,
    device: Arc<Device>,
}

unsafe impl Send for Semaphore {}
unsafe impl Sync for Semaphore {}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_semaphore(self.handle, None) }
    }
}

impl Semaphore {
    pub fn new(device: Arc<Device>) -> Result<Arc<Self>, Box<dyn Error>> {
        let create_info = vk::SemaphoreCreateInfo::default();
        let handle = unsafe { device.handle.create_semaphore(&create_info, None) }?;
        Ok(Arc::new(Self { handle, device }))
    }
}
