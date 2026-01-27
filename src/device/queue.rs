use std::{
    collections::BTreeMap,
    error::Error,
    sync::{Arc, Mutex, RwLock},
};

use ash::vk;

use crate::{device::Device, error, errors::QueueError};

pub type QueuePool = BTreeMap<QueueFamilyInfo, Vec<Arc<Mutex<Queue>>>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct QueueFamilyInfo {
    pub flags: vk::QueueFlags,
    pub index: u32,
    pub queue_count: u32,
    pub present_support: bool,
}

impl QueueFamilyInfo {}

pub struct Queue {
    pub(crate) handle: vk::Queue,
    pub device: Arc<Device>,
}

unsafe impl Send for Queue {}

impl Drop for Queue {
    fn drop(&mut self) {
        self.wait_idle().unwrap();
    }
}

impl Queue {
    pub fn submit(
        &mut self,
        submit_info: &[vk::SubmitInfo],
        fence: vk::Fence,
    ) -> Result<(), Box<dyn Error>> {
        match unsafe {
            self.device
                .handle
                .queue_submit(self.handle, submit_info, fence)
        } {
            Ok(()) => Ok(()),
            Err(e) => error!(QueueError, "cannot submit queue: {e}"),
        }
    }

    pub fn wait_idle(&mut self) -> Result<(), Box<dyn Error>> {
        match unsafe { self.device.handle.queue_wait_idle(self.handle) } {
            Err(e) => Err(Box::new(QueueError::new(format!(
                "queue wait idle error: {e}"
            )))),
            Ok(()) => Ok(()),
        }
    }

    pub fn instantiate(device: Arc<Device>) -> QueuePool {
        let mut queues = BTreeMap::new();

        device
            .physical_device
            .queue_families_info
            .iter()
            .for_each(|info| {
                let family_queues = (0..info.queue_count)
                    .map(|idx| {
                        Arc::new(Mutex::new(Queue {
                            device: device.clone(),
                            handle: unsafe { device.handle.get_device_queue(info.index, idx) },
                        }))
                    })
                    .collect::<Vec<Arc<Mutex<Queue>>>>();
                queues.insert(*info, family_queues).unwrap_or_default();
            });

        queues
    }
}
