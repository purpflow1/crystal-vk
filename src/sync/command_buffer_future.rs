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
    device::{Device, queue::Queue},
    error,
    errors::SyncError,
    sync::GpuFuture,
};

pub struct CommandBufferFuture {
    device: Arc<Device>,
    queue: Arc<Mutex<Queue>>,

    fence: vk::Fence,
    wait_semaphores: VecDeque<vk::Semaphore>,
    signal_semaphores: VecDeque<vk::Semaphore>,

    command_buffer: vk::CommandBuffer,

    submitted: bool,
    completed: bool,
    waker: Option<Waker>,
}

impl Drop for CommandBufferFuture {
    fn drop(&mut self) {
        // if self.submitted && !self.completed {
        //     let _ = unsafe {
        //         self.device
        //             .handle
        //             .wait_for_fences(&[self.fence], true, u64::MAX)
        //     };
        // }

        self.cleanup_resources();

        unsafe {
            self.device.handle.destroy_fence(self.fence, None);
        }
    }
}

impl GpuFuture for CommandBufferFuture {
    fn get_signal_semaphores(&self) -> VecDeque<vk::Semaphore> {
        self.signal_semaphores.clone()
    }

    fn set_wait_semaphores(&mut self, semaphores: VecDeque<vk::Semaphore>) {
        semaphores
            .iter()
            .for_each(|s| self.wait_semaphores.push_back(*s));
    }
}

impl Future for CommandBufferFuture {
    type Output = Result<(), Box<dyn Error>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if !self.submitted {
            if let Err(e) = self.flush() {
                self.completed = true;
                return Poll::Ready(Err(e));
            }
        }

        match self.check_completion() {
            Ok(true) => Poll::Ready(Ok(())),
            Ok(false) => {
                self.waker = Some(cx.waker().clone());
                Poll::Pending
            }
            Err(e) => Poll::Ready(Err(e)),
        }
    }
}

impl CommandBufferFuture {
    pub(crate) fn new(
        device: Arc<Device>,
        queue: Arc<Mutex<Queue>>,
        command_buffer: vk::CommandBuffer,
    ) -> Result<Box<Self>, Box<dyn Error>> {
        let fence_create_info = vk::FenceCreateInfo::default();
        let fence = match unsafe { device.handle.create_fence(&fence_create_info, None) } {
            Ok(fence) => fence,
            Err(e) => return error!(SyncError, "cannot create fence: {e}"),
        };

        let semaphore_create_info = vk::SemaphoreCreateInfo::default();

        let mut signal_semaphores = VecDeque::new();

        let signal_semaphore = unsafe {
            device
                .handle
                .create_semaphore(&semaphore_create_info, None)?
        };

        signal_semaphores.push_back(signal_semaphore);

        Ok(Box::new(Self {
            device,
            queue,
            fence,
            wait_semaphores: VecDeque::new(),
            signal_semaphores,
            command_buffer,
            submitted: false,
            completed: false,
            waker: None,
        }))
    }

    pub fn wait(&mut self) -> Result<(), Box<dyn Error>> {
        unsafe {
            if let Err(e) = self
                .device
                .handle
                .wait_for_fences(&[self.fence], true, u64::MAX)
            {
                return error!(SyncError, "error waiting for fences: {e}");
            }
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), Box<dyn Error>> {
        if self.submitted {
            return Ok(());
        }

        let command_buffers = [self.command_buffer];

        let wait_semaphores_vec: Vec<vk::Semaphore> =
            self.wait_semaphores.iter().copied().collect();
        let signal_semaphores_vec: Vec<vk::Semaphore> =
            self.signal_semaphores.iter().copied().collect();

        let wait_stages = vec![vk::PipelineStageFlags::TOP_OF_PIPE; wait_semaphores_vec.len()];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores_vec)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores_vec);

        let mut queue = self.queue.lock().unwrap();
        queue.submit(&[submit_info], self.fence)?;
        drop(queue);

        self.submitted = true;

        if let Some(waker) = self.waker.take() {
            waker.wake();
        }

        Ok(())
    }

    fn check_completion(&mut self) -> Result<bool, Box<dyn Error>> {
        if !self.submitted || self.completed {
            return Ok(self.completed);
        }

        let result = unsafe { self.device.handle.get_fence_status(self.fence) };

        match result {
            Ok(true) => {
                self.completed = true;
                self.cleanup_resources();
                Ok(true)
            }
            Ok(false) => Ok(false),
            Err(vk::Result::NOT_READY) => Ok(false),
            Err(e) => error!(SyncError, "cannot get fence status: {e}"),
        }
    }

    fn cleanup_resources(&mut self) {
        for semaphore in self.signal_semaphores.drain(..) {
            unsafe {
                self.device.handle.destroy_semaphore(semaphore, None);
            }
        }
    }
}
