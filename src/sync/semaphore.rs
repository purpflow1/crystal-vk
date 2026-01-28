use std::{error::Error, sync::Arc};

use ash::vk;

use crate::{device::Device, error, errors::SyncError};

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

        let handle = match unsafe { device.handle.create_semaphore(&create_info, None) } {
            Ok(semaphore) => semaphore,
            Err(e) => return error!(SyncError, "cannot create semaphore: {e}"),
        };

        Ok(Arc::new(Self { handle, device }))
    }
}
