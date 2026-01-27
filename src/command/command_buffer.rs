use std::{
    error::Error,
    pin::Pin,
    sync::{Arc, Mutex},
};

use ash::vk;

use crate::{
    device::{Device, queue::Queue},
    sync::CommandBufferFuture,
};

pub struct CommandBuffer {
    pub(crate) handle: vk::CommandBuffer,
    pub device: Arc<Device>,
}

impl CommandBuffer {
    pub fn execute(
        &self,
        queue: Arc<Mutex<Queue>>,
    ) -> Result<Box<CommandBufferFuture>, Box<dyn Error>> {
        let future = CommandBufferFuture::new(self.device.clone(), queue, self.handle)?;
        Ok(future)
    }
}
