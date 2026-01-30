use std::{f32::consts::PI, time::SystemTime};

use ash::vk::{self};
use crystal_vk::{
    command::command_buffer_builder::CommandBufferBuilder,
    render::{RenderTarget, swapchain::Swapchain},
    sync::SwapchainFuture,
};
use futures::executor;

use crate::vulkan_context::VulkanContext;

impl VulkanContext {
    pub fn render(&mut self, window: &winit::window::Window) {
        let delta_time = SystemTime::now().duration_since(self.last_frame).unwrap();
        self.last_frame = SystemTime::now();
        let aspect_ratio = self.extent[0] as f32 / self.extent[1] as f32;

        window.set_title(format!("FPS: {}", (1. / delta_time.as_secs_f32()) as u32).as_str());

        let camera = glam::Mat4::look_at_lh(
            glam::Vec3::new(0., 0., -1.),
            glam::Vec3::ZERO,
            glam::Vec3::new(0., 1., 0.),
        );

        let perspective = glam::Mat4::perspective_lh(PI / 3., aspect_ratio, 0.1, 100.);

        let render_camera = perspective * camera;

        let mut buffer = self.buffer_model.write().unwrap();

        let seconds = self.startup_time.elapsed().unwrap().as_secs_f32();

        let model = glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::new(0.8, 0.8, 0.8),
            glam::Quat::from_rotation_y(seconds) * glam::Quat::from_rotation_z(seconds),
            glam::Vec3::new(0., 0., 1.),
        );

        buffer[0] = render_camera * model;

        let mut buffer = self.buffer_resolution_uniform.write().unwrap();
        buffer[0] = glam::Vec2::new(self.extent[0] as f32, self.extent[1] as f32);

        drop(buffer);

        let (family_info, queues) = self
            .queues
            .iter()
            .find(|(family, _)| family.flags.contains(vk::QueueFlags::GRAPHICS))
            .unwrap();

        let suboptimal = if let Some(future) = &mut self.prev_future {
            match executor::block_on(future) {
                Ok(suboptimal) => suboptimal,
                Err(e) => {
                    dbg!(e);
                    false
                }
            }
        } else {
            false
        };

        self.prev_future = None;

        if suboptimal {
            self.swapchain = Swapchain::from_old(self.swapchain.clone(), self.extent).unwrap();

            let post_process_image = crystal_vk::image::Image::new(
                self.device.clone(),
                self.extent,
                vk::Format::R8G8B8A8_SRGB,
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

        // TODO not safe
        let mut swapchain_future =
            SwapchainFuture::new(self.device.clone(), self.swapchain.clone()).unwrap();

        // blocks until aviability
        let (image_index, _suboptimal) = match swapchain_future.acquire_next_image() {
            Ok(result) => result,
            Err(_e) => {
                dbg!(_e);
                self.prev_future = None;
                return;
            }
        };

        let queue = queues[0].clone();

        let builder = CommandBufferBuilder::new(
            self.command_allocator.clone(),
            family_info.index,
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        )
        .unwrap()
        .begin_render_pass(self.post_process_render_target.clone(), 0)
        .unwrap()
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
        .bind_index_buffer(self.buffer_ind.clone())
        .bind_descriptor_sets(
            self.pipeline.pipeline_layout.clone(),
            0,
            vec![self.per_object_descriptor_set.clone()],
        )
        .draw_indexed(36, 1, 0, 0, 0)
        .end_render_pass()
        .begin_render_pass(self.swapchain_render_target.clone(), image_index)
        .unwrap()
        .bind_pipeline(self.post_process_pipeline.clone())
        .bind_vertex_buffer(self.buffer_vert.clone())
        .bind_index_buffer(self.buffer_ind.clone())
        .bind_descriptor_sets(
            self.post_process_pipeline.pipeline_layout.clone(),
            0,
            vec![self.post_process_descriptor_set.clone()],
        )
        .draw_indexed(6, 1, 36, 8, 0)
        .end_render_pass();

        executor::block_on(swapchain_future).unwrap();

        let command_buffer_future = builder
            .build(queue)
            .unwrap()
            .then_present(self.swapchain.clone(), image_index)
            .unwrap();

        // command_buffer_future.flush().unwrap();

        self.prev_future = Some(Box::pin(command_buffer_future));
    }
}
