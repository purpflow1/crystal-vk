use std::{error::Error, f32::consts::PI, sync::atomic::Ordering, time::Duration};

use crystal_vk::{
    command::command_buffer_builder::CommandBufferBuilder,
    render::{RenderTarget, swapchain::Swapchain},
    sync::SwapchainFuture,
    vk,
};
use futures::executor;

use crate::{vulkan_context::VulkanContext, watcher::watcher};

impl VulkanContext {
    /// # Safety
    /// It's better to `unwrap()` here, so it's easier to debug
    pub fn render(&mut self, window: &winit::window::Window) -> Result<(), Box<dyn Error>> {
        if self.first_frame {
            self.watcher = {
                let stop_flag = self.stop_flag.clone();
                let heartbeat = self.heartbeat.clone();
                Some(watcher(stop_flag, heartbeat, Duration::from_secs(1)))
            };

            self.first_frame = false;
        }

        self.heartbeat.store(true, Ordering::Relaxed);
        self.timeline.frame_begin();

        let aspect_ratio = self.extent[0] as f32 / self.extent[1] as f32;
        window.set_title(
            format!(
                "FPS: [avg {} max {} min {}]",
                (1. / self.timeline.average_delta_time_last_second) as u32,
                (1. / self.timeline.min_delta) as u32,
                (1. / self.timeline.max_delta) as u32
            )
            .as_str(),
        );

        let camera = glam::Mat4::look_at_lh(
            glam::Vec3::new(0., 0., -1.),
            glam::Vec3::ZERO,
            glam::Vec3::new(0., 1., 0.),
        );

        let perspective = glam::Mat4::perspective_lh(PI / 3., aspect_ratio, 0.1, 100.);

        let render_camera = perspective * camera;

        let mut lock = self.buffer_model.write().unwrap();

        let seconds = self.timeline.startup_time.elapsed().unwrap().as_secs_f32();

        let model = glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::new(0.8, 0.8, 0.8),
            glam::Quat::from_rotation_y(seconds) * glam::Quat::from_rotation_z(seconds),
            glam::Vec3::new(0., 0., 1.),
        );

        let eye = (render_camera * model).to_cols_array();

        let size = lock.info.size;
        let memory = lock.bind_memory(0..size).unwrap();
        memory.copy_from_slice(bytemuck::cast_slice(&eye));

        let mut lock = self.buffer_resolution_uniform.write().unwrap();
        let memory = lock.bind_memory(0..8).unwrap();
        memory.copy_from_slice(bytemuck::cast_slice(&[
            self.extent[0] as f32,
            self.extent[1] as f32,
        ]));

        let (family_info, queues) = self
            .queues
            .iter()
            .find(|(family, _)| family.flags.contains(vk::QueueFlags::GRAPHICS))
            .unwrap();

        let (suboptimal, mut out_of_date) = if let Some(future) = &mut self.prev_future {
            match executor::block_on(future) {
                Ok(suboptimal) => (suboptimal, false),
                Err(e) => {
                    dbg!(e);
                    (false, true)
                }
            }
        } else {
            (false, false)
        };

        self.prev_future = None;

        if self.extent_changed {
            out_of_date = true;
            self.extent_changed = false
        }

        let queue = queues[0].clone();

        if suboptimal || out_of_date {
            let mut lock = queue.lock().unwrap();
            lock.wait_idle().unwrap();
            drop(lock);

            self.swapchain = Swapchain::from_old(self.swapchain.clone(), self.extent).unwrap();

            let post_process_image = crystal_vk::image::Image::new(
                self.device.clone(),
                self.extent,
                vk::Format::R8G8B8A8_SRGB,
                vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            )
            .unwrap();

            let mut lock = self.post_process_descriptor_set.lock().unwrap();
            lock.bind_combined_image_sampler(
                post_process_image.clone(),
                self.post_process_sampler.clone(),
                1,
                0,
                1,
            )
            .unwrap();

            drop(lock);

            self.post_process_render_target =
                RenderTarget::new(self.device.clone(), vec![post_process_image.clone()], 4)
                    .unwrap();

            self.swapchain_render_target = RenderTarget::new(
                self.device.clone(),
                self.swapchain.image_sequence.clone(),
                4,
            )
            .unwrap();
        }

        let mut swapchain_future = SwapchainFuture::new(self.swapchain.clone()).unwrap();

        let (image_index, out_of_date) = match swapchain_future.flush() {
            Ok(result) => (result.0, false),
            Err(_e) => {
                dbg!(_e);
                (0, true)
            }
        };

        if out_of_date {
            executor::block_on(swapchain_future).unwrap();
            self.swapchain = Swapchain::new(queue.clone(), self.extent, true).unwrap();

            let post_process_image = crystal_vk::image::Image::new(
                self.device.clone(),
                self.extent,
                vk::Format::R8G8B8A8_SRGB,
                vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            )
            .unwrap();

            let mut lock = self.post_process_descriptor_set.lock().unwrap();
            lock.bind_combined_image_sampler(
                post_process_image.clone(),
                self.post_process_sampler.clone(),
                1,
                0,
                1,
            )
            .unwrap();

            drop(lock);

            self.post_process_render_target =
                RenderTarget::new(self.device.clone(), vec![post_process_image.clone()], 4)
                    .unwrap();

            self.swapchain_render_target = RenderTarget::new(
                self.device.clone(),
                self.swapchain.image_sequence.clone(),
                4,
            )
            .unwrap();

            return Ok(());
        }

        let builder = CommandBufferBuilder::new(self.command_allocator.clone(), family_info.index)
            .unwrap()
            .begin_render_pass(self.post_process_render_target.clone(), 0)
            .bind_viewport_and_scissor(
                vec![vk::Viewport {
                    width: self.extent[0] as f32,
                    height: self.extent[1] as f32,
                    ..Default::default()
                }],
                vec![vk::Rect2D {
                    extent: vk::Extent2D {
                        width: self.extent[0],
                        height: self.extent[1],
                    },
                    ..Default::default()
                }],
            )
            .bind_pipeline(self.pipeline.clone())
            .bind_vertex_buffer(self.buffer_vert.clone())
            .bind_index_buffer(self.buffer_ind.clone(), vk::IndexType::UINT16)
            .bind_descriptor_sets(0, vec![self.per_object_descriptor_set.clone()])
            .draw_indexed(36, 1, 0, 0, 0)
            .end_render_pass()
            .begin_render_pass(self.swapchain_render_target.clone(), image_index)
            .bind_pipeline(self.post_process_pipeline.clone())
            .bind_descriptor_sets(0, vec![self.post_process_descriptor_set.clone()])
            .draw_indexed(6, 1, 36, 8, 0)
            .end_render_pass();

        // blocking on swapchain to complete acquiring
        executor::block_on(swapchain_future).unwrap();

        let mut command_buffer_future = builder
            .build(queue)
            .unwrap()
            .then_present(self.swapchain.clone(), image_index)
            .unwrap();

        // submitting command buffer
        command_buffer_future.flush().unwrap();

        self.prev_future = Some(Box::pin(command_buffer_future));

        Ok(())
    }
}
