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
    error,
    errors::{QueueError, SyncError},
    render::swapchain::Swapchain,
    sync::GpuFuture,
};

pub struct PresentFuture {
    device: Arc<Device>,
    swapchain: Arc<Swapchain>,

    wait_semaphores: VecDeque<vk::Semaphore>,
    fence: vk::Fence,

    submitted: bool,
    completed: bool,
    waker: Option<Waker>,

    suboptimal: bool,
    image_index: u32,
}

impl Drop for PresentFuture {
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
        }
    }
}

impl Future for PresentFuture {
    type Output = Result<(u32, bool), Box<dyn Error>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let image_index = self.image_index;

        if !self.submitted {
            if let Err(e) = self.present(image_index) {
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

impl GpuFuture for PresentFuture {
    fn get_signal_semaphores(&self) -> VecDeque<vk::Semaphore> {
        VecDeque::with_capacity(0)
    }

    fn set_wait_semaphores(&mut self, semaphores: VecDeque<vk::Semaphore>) {
        self.wait_semaphores = semaphores
    }
}

impl PresentFuture {
    pub fn new(
        device: Arc<Device>,
        swapchain: Arc<Swapchain>,
    ) -> Result<Box<Self>, Box<dyn Error>> {
        let fence_create_info = vk::FenceCreateInfo::default();

        let fence = unsafe { device.handle.create_fence(&fence_create_info, None)? };

        Ok(Box::new(Self {
            device,
            swapchain,
            wait_semaphores: VecDeque::new(),
            fence,
            submitted: false,
            completed: false,
            waker: None,
            suboptimal: false,
            image_index: u32::MAX,
        }))
    }

    pub fn present(&mut self, image_index: u32) -> Result<bool, Box<dyn Error>> {
        self.image_index = image_index;
        if self.submitted {
            return Ok(self.suboptimal);
        }

        let wait_semaphores_vec: Vec<vk::Semaphore> =
            self.wait_semaphores.iter().copied().collect();
        let swapchains = [self.swapchain.swapchain_khr];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores_vec)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let queue_lock = self.swapchain.present_queue.lock().unwrap();

        self.suboptimal = match unsafe {
            self.swapchain
                .swapchain
                .queue_present(queue_lock.handle, &present_info)
        } {
            Ok(suboptimal) => suboptimal,
            Err(e) => return error!(QueueError, "cannot queue_present: {e}"),
        };

        self.submitted = true;

        if let Some(waker) = self.waker.take() {
            waker.wake();
        }

        Ok(self.suboptimal)
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
