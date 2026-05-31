use std::{
    collections::BTreeMap,
    error::Error,
    sync::{Arc, Mutex},
};

use ash::vk;

use crate::{device::Device, render::swapchain::Swapchain};

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
    handle: vk::Queue,
    pub device: Arc<Device>,
}

unsafe impl Send for Queue {}

impl Queue {
    pub fn present(
        &self,
        present_info: &vk::PresentInfoKHR,
        swapchain: Arc<Swapchain>,
    ) -> Result<bool, Box<dyn Error>> {
        Ok(unsafe { swapchain.swapchain.queue_present(self.handle, present_info) }?)
    }

    pub fn submit(
        &mut self,
        submit_info: &[vk::SubmitInfo],
        fence: vk::Fence,
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            self.device
                .handle
                .queue_submit(self.handle, submit_info, fence)?
        }

        Ok(())
    }

    pub fn wait_idle(&mut self) -> Result<(), Box<dyn Error>> {
        Ok(unsafe { self.device.handle.queue_wait_idle(self.handle) }?)
    }

    pub fn instantiate(device: Arc<Device>) -> QueuePool {
        device
            .physical_device
            .info
            .queue_families_info
            .iter()
            .filter_map(|info| {
                let family_queues = (0..info.queue_count)
                    .map(|idx| {
                        Arc::new(Mutex::new(Queue {
                            device: device.clone(),
                            handle: unsafe { device.handle.get_device_queue(info.index, idx) },
                        }))
                    })
                    .collect::<Vec<Arc<Mutex<Queue>>>>();
                if info.queue_count > 0 {
                    Some((*info, family_queues))
                } else {
                    None
                }
            })
            .collect()
    }
}
