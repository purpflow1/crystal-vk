pub mod binding;
pub mod command_buffer_builder;

use std::{
    collections::BTreeMap,
    error::Error,
    sync::{Arc, Mutex},
};

use ash::vk;

use crate::{
    device::{Device, queue::QueuePool},
    error,
    errors::CommandError,
};

/// Allocator per thread
pub struct CommandBufferAllocator {
    pub(crate) pools: BTreeMap<u32, Arc<Mutex<vk::CommandPool>>>,
    pub device: Arc<Device>,
}

impl Drop for CommandBufferAllocator {
    fn drop(&mut self) {
        self.pools.iter().for_each(|(_, pool)| {
            let lock = pool.lock().unwrap();
            unsafe { self.device.handle.destroy_command_pool(*lock, None) };
        });
    }
}

impl CommandBufferAllocator {
    pub fn new(queue_pool: QueuePool) -> Result<Arc<Self>, Box<dyn Error>> {
        let mut pools = BTreeMap::new();

        let mut device = None;

        for (family_info, queues) in queue_pool.iter() {
            let create_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(family_info.index);

            let queue = queues[0].lock().unwrap();

            device = Some(queue.device.clone());

            let command_pool =
                match unsafe { queue.device.handle.create_command_pool(&create_info, None) } {
                    Ok(command_pool) => command_pool,
                    Err(e) => {
                        return error!(CommandError, "cannot create command pool: {e}");
                    }
                };

            pools.insert(family_info.index, Arc::new(Mutex::new(command_pool)));
        }

        Ok(Arc::new(Self {
            pools,
            device: device.unwrap(),
        }))
    }
}
