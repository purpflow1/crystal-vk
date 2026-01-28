mod command_buffer_future;
mod present_future;
mod semaphore;
mod swapchain_future;

use std::{collections::VecDeque, sync::Arc};

pub use command_buffer_future::*;
pub use present_future::*;
pub(crate) use semaphore::*;
pub use swapchain_future::*;

pub trait GpuFuture: Future {
    fn set_wait_semaphores(&mut self, semaphores: VecDeque<Arc<Semaphore>>);
    fn get_signal_semaphores(&self) -> VecDeque<Arc<Semaphore>>;

    fn sync_with_present(&self, after: &mut Box<PresentFuture>)
    where
        Self: Sized,
    {
        after.set_wait_semaphores(self.get_signal_semaphores());
    }

    fn sync_with(&self, after: &mut Self)
    where
        Self: Sized,
    {
        after.set_wait_semaphores(self.get_signal_semaphores());
    }
}
