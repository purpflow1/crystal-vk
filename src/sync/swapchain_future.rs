use std::{
    collections::VecDeque,
    error::Error,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, Waker},
};

use ash::vk;

use crate::{
    device::Device,
    render::swapchain::Swapchain,
    sync::{GpuFuture, Semaphore},
};

pub struct SwapchainFuture {
    device: Arc<Device>,
    swapchain: Arc<Swapchain>,

    signal_semaphore: Arc<Semaphore>,
    fence: vk::Fence,

    submitted: bool,
    completed: bool,
    waker: Option<Waker>,

    suboptimal: bool,
    image_index: u32,
}

impl Drop for SwapchainFuture {
    fn drop(&mut self) {
        if self.submitted && !self.completed {
            let _ = unsafe {
                self.device
                    .handle
                    .wait_for_fences(&[self.fence], true, u64::MAX)
            };
        }
        unsafe {
            self.device.handle.destroy_fence(self.fence, None);
        }
    }
}

impl Future for SwapchainFuture {
    type Output = Result<(), Box<dyn Error>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if !self.submitted {
            match self.flush() {
                Ok(_) => {
                    self.submitted = true;
                    self.waker = Some(cx.waker().clone());
                }
                Err(e) => {
                    self.completed = true;
                    return Poll::Ready(Err(e));
                }
            }
        }

        match self.check_completion() {
            Ok(true) => {
                self.completed = true;
                Poll::Ready(Ok(()))
            }
            Ok(false) => {
                self.waker = Some(cx.waker().clone());
                Poll::Pending
            }
            Err(e) => {
                self.completed = true;
                Poll::Ready(Err(e))
            }
        }
    }
}

impl GpuFuture for SwapchainFuture {
    fn get_signal_semaphores(&self) -> VecDeque<Arc<Semaphore>> {
        let mut deque = VecDeque::new();
        deque.push_back(self.signal_semaphore.clone());
        deque
    }

    fn set_wait_semaphores(&mut self, _semaphores: VecDeque<Arc<Semaphore>>) {}
}

impl SwapchainFuture {
    pub(crate) fn new(
        device: Arc<Device>,
        swapchain: Arc<Swapchain>,
    ) -> Result<Box<Self>, Box<dyn Error>> {
        let signal_semaphore = Semaphore::new(device.clone())?;
        let fence_create_info = vk::FenceCreateInfo::default();
        let fence = unsafe { device.handle.create_fence(&fence_create_info, None)? };

        Ok(Box::new(Self {
            device,
            swapchain,
            signal_semaphore,
            fence,
            submitted: false,
            completed: false,
            waker: None,
            suboptimal: false,
            image_index: u32::MAX,
        }))
    }

    pub fn flush(&mut self) -> Result<(u32, bool), Box<dyn Error>> {
        if self.submitted {
            return Ok((self.image_index, self.suboptimal));
        }

        let (image_index, suboptimal) = match unsafe {
            self.swapchain.swapchain.acquire_next_image(
                self.swapchain.swapchain_khr,
                u64::MAX,
                self.signal_semaphore.handle,
                self.fence,
            )
        } {
            Ok(result) => result,
            Err(vk::Result::SUBOPTIMAL_KHR) => (self.image_index, true),
            Err(e) => return Err(e.into()),
        };

        self.image_index = image_index;
        self.suboptimal = suboptimal;

        self.submitted = true;

        Ok((image_index, suboptimal))
    }

    fn check_completion(&mut self) -> Result<bool, Box<dyn Error>> {
        if !self.submitted || self.completed {
            return Ok(self.completed);
        }

        let result = unsafe { self.device.handle.get_fence_status(self.fence) };

        if let Some(waker) = self.waker.take() {
            waker.wake();
        }

        match result {
            Ok(true) => {
                self.completed = true;
                Ok(true)
            }
            Ok(false) | Err(vk::Result::NOT_READY) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }
}
