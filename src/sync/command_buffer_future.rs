use std::{
    collections::VecDeque,
    error::Error,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
};

use ash::vk;
use std::sync::MutexGuard;

use crate::{
    command::command_buffer_builder::CommandBufferBuilder,
    device::{Device, queue::Queue},
    render::swapchain::Swapchain,
    sync::{GpuFuture, Semaphore},
};

/// # Safety
/// `Queue` is locked the whole time, this future exists
pub struct CommandBufferFuture<'a> {
    device: Arc<Device>,

    fence: vk::Fence,
    wait_semaphores: VecDeque<Arc<Semaphore>>,
    signal_semaphores: VecDeque<Arc<Semaphore>>,

    builder: Box<CommandBufferBuilder>,

    suboptimal: bool,

    submitted: bool,
    completed: bool,

    present: bool,
    image_index: u32,
    swapchain: Option<Arc<Swapchain>>,

    locked_queue: MutexGuard<'a, Queue>,

    waker: Option<Waker>,
}

unsafe impl<'a> Send for CommandBufferFuture<'a> {}

impl<'a> Drop for CommandBufferFuture<'a> {
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

impl<'a> GpuFuture for CommandBufferFuture<'a> {
    fn get_signal_semaphores(&self) -> VecDeque<Arc<Semaphore>> {
        self.signal_semaphores.clone()
    }

    fn set_wait_semaphores(&mut self, semaphores: VecDeque<Arc<Semaphore>>) {
        semaphores
            .iter()
            .for_each(|s| self.wait_semaphores.push_back(s.clone()));
    }
}

impl<'a> Future for CommandBufferFuture<'a> {
    type Output = Result<bool, Box<dyn Error>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if !self.submitted {
            match self.flush() {
                Ok(()) => {
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
                Poll::Ready(Ok(self.suboptimal))
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

impl<'a> CommandBufferFuture<'a> {
    pub fn then_present(
        mut self: Box<Self>,
        swapchain: Arc<Swapchain>,
        image_index: u32,
    ) -> Result<Box<Self>, Box<dyn Error>> {
        self.present = true;
        self.swapchain = Some(swapchain);
        self.image_index = image_index;
        Ok(self)
    }

    pub fn present(&self) -> Result<bool, Box<dyn Error>> {
        let wait_semaphores_vec: Vec<vk::Semaphore> =
            self.signal_semaphores.iter().map(|s| s.handle).collect();
        let swapchains = [self.swapchain.as_ref().unwrap().swapchain_khr];
        let image_indices = [self.image_index];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores_vec)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let suboptimal = self
            .locked_queue
            .present(&present_info, self.swapchain.clone().unwrap())?;

        Ok(suboptimal)
    }

    pub(crate) fn new(
        bulder: Box<CommandBufferBuilder>,
        queue: Arc<Mutex<Queue>>,
    ) -> Result<Box<Self>, Box<dyn Error>> {
        let device = bulder.command_buffer_allocator.device.clone();

        let fence_create_info = vk::FenceCreateInfo::default();
        let fence = unsafe { device.handle.create_fence(&fence_create_info, None) }?;

        let mut signal_semaphores = VecDeque::new();
        let signal_semaphore = Semaphore::new(device.clone())?;
        signal_semaphores.push_back(signal_semaphore);

        Ok(Box::new(Self {
            device,
            fence,
            wait_semaphores: VecDeque::new(),
            signal_semaphores,
            builder: bulder,

            suboptimal: false,

            present: false,
            image_index: u32::MAX,
            swapchain: None,

            locked_queue: unsafe { std::mem::transmute(queue.lock().unwrap()) },

            submitted: false,
            completed: false,
            waker: None,
        }))
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

    pub fn flush(&mut self) -> Result<(), Box<dyn Error>> {
        if self.submitted {
            return Ok(());
        }

        let command_buffers = [self.builder.handle];

        let wait_semaphores_vec: Vec<vk::Semaphore> =
            self.wait_semaphores.iter().map(|s| s.handle).collect();
        let signal_semaphores_vec: Vec<vk::Semaphore> =
            self.signal_semaphores.iter().map(|s| s.handle).collect();

        let wait_stages = vec![vk::PipelineStageFlags::TOP_OF_PIPE; wait_semaphores_vec.len()];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores_vec)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores_vec);

        self.locked_queue.submit(&[submit_info], self.fence)?;

        let suboptimal = if self.present { self.present()? } else { false };
        self.suboptimal = suboptimal;

        self.submitted = true;

        Ok(())
    }

    pub fn wait(&mut self) -> Result<(), Box<dyn Error>> {
        unsafe {
            self.device
                .handle
                .wait_for_fences(&[self.fence], true, u64::MAX)?
        }
        Ok(())
    }
}
