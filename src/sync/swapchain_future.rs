use std::{
    collections::VecDeque,
    error::Error,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
    u64,
};

use ash::vk;

use crate::{
    command::command_buffer::CommandBuffer,
    device::{Device, queue::Queue},
    error,
    errors::{CommandError, SwapchainOutOfDate, SyncError},
    render::swapchain::Swapchain,
    sync::GpuFuture,
};

pub struct SwapchainFuture {
    device: Arc<Device>,
    swapchain: Arc<Swapchain>,

    signal_semaphore: vk::Semaphore,
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
            self.swapchain
                .present_queue
                .lock()
                .unwrap()
                .wait_idle()
                .unwrap();
        }
        unsafe {
            self.device.handle.destroy_fence(self.fence, None);
            self.device
                .handle
                .destroy_semaphore(self.signal_semaphore, None);
        }
    }
}

impl Future for SwapchainFuture {
    type Output = Result<(u32, bool), Box<dyn Error>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if !self.submitted {
            if let Err(e) = self.acquire_next_image() {
                self.completed = true;
                return Poll::Ready(Err(e));
            }
        }

        match self.check_completion() {
            Ok(true) => Poll::Ready(Ok((self.image_index, self.suboptimal))),
            Ok(false) => {
                self.waker = Some(cx.waker().clone());
                Poll::Pending
            }
            Err(e) => Poll::Ready(Err(e)),
        }
    }
}

impl GpuFuture for SwapchainFuture {
    fn get_signal_semaphores(&self) -> VecDeque<vk::Semaphore> {
        let mut deque = VecDeque::new();
        deque.push_back(self.signal_semaphore);
        deque
    }

    fn set_wait_semaphores(&mut self, _semaphores: VecDeque<vk::Semaphore>) {}
}

impl SwapchainFuture {
    pub fn new(
        device: Arc<Device>,
        swapchain: Arc<Swapchain>,
    ) -> Result<Box<Self>, Box<dyn Error>> {
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();

        let signal_semaphore = unsafe {
            device
                .handle
                .create_semaphore(&semaphore_create_info, None)?
        };

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

    pub fn acquire_next_image(&mut self) -> Result<(u32, bool), Box<dyn Error>> {
        if self.submitted {
            return Ok((self.image_index, self.suboptimal));
        }

        let (image_index, suboptimal) = match unsafe {
            self.swapchain.swapchain.acquire_next_image(
                self.swapchain.swapchain_khr,
                u64::MAX,
                self.signal_semaphore,
                vk::Fence::null(),
            )
        } {
            Ok(result) => result,
            Err(vk::Result::SUBOPTIMAL_KHR) => (self.image_index, true),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                return error!(SwapchainOutOfDate, "out of date");
            }
            Err(e) => return error!(SyncError, "cannot acquire_next_image: {e}"),
        };

        self.image_index = image_index;
        self.suboptimal = suboptimal;

        self.submitted = true;

        if let Some(waker) = self.waker.take() {
            waker.wake();
        }

        Ok((image_index, suboptimal))
    }

    fn check_completion(&mut self) -> Result<bool, Box<dyn Error>> {
        if !self.submitted || self.completed {
            return Ok(self.completed);
        }

        let result = unsafe { self.device.handle.get_fence_status(self.fence) };

        match result {
            Ok(true) => {
                self.completed = true;
                Ok(true)
            }
            Ok(false) => Ok(false),
            Err(vk::Result::NOT_READY) => Ok(false),
            Err(e) => error!(SyncError, "cannot get fence status: {e}"),
        }
    }
}
