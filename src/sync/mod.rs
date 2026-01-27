mod command_buffer_future;
mod present_future;
mod swapchain_future;

use std::{collections::VecDeque, error::Error};

use ash::vk;
pub use command_buffer_future::*;
pub use present_future::*;
pub use swapchain_future::*;

pub trait GpuFuture: Future {
    fn set_wait_semaphores(&mut self, semaphores: VecDeque<vk::Semaphore>);
    fn get_signal_semaphores(&self) -> VecDeque<vk::Semaphore>;

    fn sync_with_present(self: &Box<Self>, after: &mut Box<PresentFuture>)
    where
        Self: Sized,
    {
        after.set_wait_semaphores(self.get_signal_semaphores());
    }

    fn sync_with(self: &Box<Self>, after: &mut Box<Self>)
    where
        Self: Sized,
    {
        after.set_wait_semaphores(self.get_signal_semaphores());
    }
}
