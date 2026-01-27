use std::{
    error::Error,
    marker::PhantomData,
    sync::{Arc, Mutex, RwLock},
};

use ash::vk;

use crate::{
    buffer::Buffer,
    command::{CommandBufferAllocator, command_buffer::CommandBuffer},
    device::Device,
    error,
    errors::CommandError,
    pipeline::{
        Pipeline,
        descriptor::{
            descriptor_set_layout::descriptor_set::DescriptorSet, layout::PipelineLayout,
        },
    },
    render::{RenderTarget, framebuffer::FramebufferPool},
};

#[derive(Default)]
pub struct CommandBufferBuilderInfo {
    last_pipeline_bind_point: vk::PipelineBindPoint,
}

/// # Safety
/// Everything is boxed
pub struct CommandBufferBuilder {
    handle: vk::CommandBuffer,
    command_buffer_allocator: Arc<CommandBufferAllocator>,
    info: CommandBufferBuilderInfo,
}

impl CommandBufferBuilder {
    pub fn build(self: Box<Self>) -> Result<Box<CommandBuffer>, Box<dyn Error>> {
        match unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .end_command_buffer(self.handle)
        } {
            Ok(()) => (),
            Err(e) => {
                return error!(CommandError, "cannot build command buffer: {e}");
            }
        }

        self.validate()?;
        Ok(Box::new(CommandBuffer {
            handle: self.handle,
            device: self.command_buffer_allocator.device.clone(),
        }))
    }

    fn validate(&self) -> Result<(), Box<dyn Error>> {
        // TODO
        Ok(())
    }

    pub fn new(
        command_buffer_allocator: Arc<CommandBufferAllocator>,
        queue_family_index: u32,
        flags: vk::CommandBufferUsageFlags,
    ) -> Result<Box<Self>, Box<dyn Error>> {
        let command_pool = command_buffer_allocator
            .pools
            .get(&queue_family_index)
            .unwrap()
            .clone();
        let command_pool_lock = command_pool.lock().unwrap();

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(*command_pool_lock)
            .command_buffer_count(1);

        let command_buffer = match unsafe {
            command_buffer_allocator
                .device
                .handle
                .allocate_command_buffers(&alloc_info)
        } {
            Ok(buffers) => buffers[0],
            Err(e) => {
                return error!(CommandError, "cannot allocate command buffer: {e}");
            }
        };

        let begin_info = vk::CommandBufferBeginInfo::default().flags(flags);

        match unsafe {
            command_buffer_allocator
                .device
                .handle
                .begin_command_buffer(command_buffer, &begin_info)
        } {
            Ok(_) => (),
            Err(e) => {
                return error!(CommandError, "cannot begin command buffer: {e}");
            }
        }

        Ok(Box::new(Self {
            handle: command_buffer,
            command_buffer_allocator,
            info: Default::default(),
        }))
    }

    pub fn draw_indexed(self: Box<Self>, index_count: u32) -> Box<Self> {
        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .cmd_draw_indexed(self.handle, index_count, 1, 0, 0, 0);
        }
        self
    }

    pub fn bind_descriptor_sets<DT, PT>(
        self: Box<Self>,
        pipeline: Arc<Pipeline<PT>>,
        first_set: u32,
        descriptor_sets: Vec<Arc<Mutex<DescriptorSet<DT>>>>,
    ) -> Box<Self> {
        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .cmd_bind_descriptor_sets(
                    self.handle,
                    self.info.last_pipeline_bind_point,
                    pipeline.pipeline_layout.handle,
                    first_set,
                    &descriptor_sets
                        .iter()
                        .map(|set| set.lock().unwrap().handle)
                        .collect::<Vec<_>>(),
                    &[],
                );
        }
        self
    }

    pub fn bind_index_buffer<T>(self: Box<Self>, buffer: Arc<RwLock<Buffer<T>>>) -> Box<Self> {
        let buffer_lock = buffer.read().unwrap();
        let buffer_raw = buffer_lock.as_raw();
        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .cmd_bind_index_buffer(self.handle, buffer_raw, 0, vk::IndexType::UINT16);
        };

        self
    }

    pub fn bind_vertex_buffer<T>(self: Box<Self>, buffer: Arc<RwLock<Buffer<T>>>) -> Box<Self> {
        unsafe {
            let buffer_lock = buffer.read().unwrap();
            let buffer_raw = buffer_lock.as_raw();

            self.command_buffer_allocator
                .device
                .handle
                .cmd_bind_vertex_buffers(self.handle, 0, &[buffer_raw], &[0]);
        };

        self
    }

    pub fn bind_pipeline<T>(
        mut self: Box<Self>,
        pipeline: Arc<Pipeline<T>>,
        pipeline_bind_point: vk::PipelineBindPoint,
    ) -> Box<Self> {
        self.info.last_pipeline_bind_point = pipeline_bind_point;

        let viewports = &pipeline.info.viewports;
        let scissors = &pipeline.info.scissors;

        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .cmd_set_viewport(self.handle, 0, viewports);
            self.command_buffer_allocator
                .device
                .handle
                .cmd_set_scissor(self.handle, 0, scissors);
            self.command_buffer_allocator
                .device
                .handle
                .cmd_bind_pipeline(self.handle, pipeline_bind_point, pipeline.handle)
        };

        self
    }

    pub fn end_render_pass(self: Box<Self>) -> Box<Self> {
        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .cmd_end_render_pass(self.handle)
        };

        self
    }

    pub fn begin_render_pass(
        self: Box<Self>,
        render_target: Arc<RenderTarget>,
        image_index: u32,
    ) -> Result<Box<Self>, Box<dyn Error>> {
        let color = 0.3f32;
        let clear_value_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [color, color, color, 1.],
            },
        };

        let clear_value_stencil = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue::default().depth(1.).stencil(0),
        };

        let clear_values = &[clear_value_color, clear_value_stencil];

        let framebuffer_lock = render_target.framebuffer.read().unwrap();
        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(framebuffer_lock.render_pass.handle)
            .framebuffer(framebuffer_lock.get_framebuffer(image_index))
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default().x(0).y(0),
                extent: vk::Extent2D {
                    width: framebuffer_lock.attachments[0].info.extent[0],
                    height: framebuffer_lock.attachments[0].info.extent[1],
                },
            })
            .clear_values(clear_values);

        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .cmd_begin_render_pass(
                    self.handle,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                )
        }

        Ok(self)
    }
}
