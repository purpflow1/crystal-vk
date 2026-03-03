use std::{
    any::Any,
    collections::VecDeque,
    error::Error,
    marker::PhantomData,
    sync::{Arc, Mutex, RwLock},
};

use ash::vk;

use crate::{
    buffer::{AnyBuffer, Buffer, InderectBuffer, IndexBuffer, VertexBuffer},
    command::CommandBufferAllocator,
    device::queue::Queue,
    image::Image,
    pipeline::{Pipeline, descriptor::descriptor_set_layout::descriptor_set::DescriptorSet},
    render::RenderTarget,
    sync::CommandBufferFuture,
};

pub trait BuilderState {}
pub trait PipelineBoundState: BuilderState {}
pub trait RenderPassBound: BuilderState {}
pub trait Buildable: BuilderState {}

pub struct Idle;
impl BuilderState for Idle {}
impl Buildable for Idle {}

pub struct PipelineBound;
impl BuilderState for PipelineBound {}
impl PipelineBoundState for PipelineBound {}
impl Buildable for PipelineBound {}

pub struct InRenderPass;
impl BuilderState for InRenderPass {}
impl RenderPassBound for InRenderPass {}

pub struct InRenderPassWithPipeline;
impl BuilderState for InRenderPassWithPipeline {}
impl PipelineBoundState for InRenderPassWithPipeline {}
impl RenderPassBound for InRenderPassWithPipeline {}

pub struct CommandBufferBuilder<State: BuilderState = Idle> {
    pub(crate) handle: vk::CommandBuffer,
    pub(crate) command_buffer_allocator: Arc<CommandBufferAllocator>,

    bindings: VecDeque<Arc<dyn Any + Send + Sync>>,

    _state: PhantomData<State>,
}

impl<State: Buildable> CommandBufferBuilder<State> {
    pub fn build(
        self,
        queue: Arc<Mutex<Queue>>,
    ) -> Result<Box<CommandBufferFuture>, Box<dyn Error>> {
        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .end_command_buffer(self.handle)
        }?;

        let new = Box::new(CommandBufferBuilder {
            handle: self.handle,
            command_buffer_allocator: self.command_buffer_allocator,
            bindings: self.bindings,
            _state: PhantomData,
        });

        CommandBufferFuture::new(new, queue)
    }
}

impl CommandBufferBuilder<Idle> {
    pub fn new(
        command_buffer_allocator: Arc<CommandBufferAllocator>,
        queue_family_index: u32,
    ) -> Result<Self, Box<dyn Error>> {
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

        let command_buffer = unsafe {
            command_buffer_allocator
                .device
                .handle
                .allocate_command_buffers(&alloc_info)
        }?[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            command_buffer_allocator
                .device
                .handle
                .begin_command_buffer(command_buffer, &begin_info)
        }?;

        Ok(Self {
            handle: command_buffer,
            command_buffer_allocator,
            bindings: VecDeque::new(),
            _state: PhantomData,
        })
    }

    pub fn begin_render_pass(
        self,
        render_target: Arc<RenderTarget>,
        image_index: u32,
    ) -> CommandBufferBuilder<InRenderPass> {
        let new = Box::new(CommandBufferBuilder {
            handle: self.handle,
            command_buffer_allocator: self.command_buffer_allocator,
            bindings: self.bindings,
            _state: PhantomData,
        });
        new.begin_render_pass_in(render_target, image_index)
    }

    pub fn bind_pipeline(self, pipeline: Arc<Pipeline>) -> CommandBufferBuilder<PipelineBound> {
        let new = CommandBufferBuilder {
            handle: self.handle,
            command_buffer_allocator: self.command_buffer_allocator,
            bindings: self.bindings,
            _state: PhantomData,
        };
        new.bind_pipeline_in(pipeline)
    }
}

impl<State: PipelineBoundState> CommandBufferBuilder<State> {
    pub fn push_constants(
        self,
        stages: vk::ShaderStageFlags,
        offset: u32,
        constants: &[u8],
    ) -> Self {
        let pipeline: Arc<Pipeline> = self
            .bindings
            .iter()
            .rfind(|item| Arc::clone(item).downcast::<Pipeline>().is_ok())
            .map(|item| Arc::clone(item).downcast().unwrap())
            .unwrap();

        let layout = pipeline.pipeline_layout.clone();

        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .cmd_push_constants(self.handle, layout.handle, stages, offset, constants)
        }

        self
    }

    pub fn bind_descriptor_sets(
        mut self,
        first_set: u32,
        descriptor_sets: Vec<Arc<Mutex<DescriptorSet>>>,
    ) -> Self {
        for descriptor_set in descriptor_sets.iter() {
            self.bindings.push_back(descriptor_set.clone());
        }

        let pipeline: Arc<Pipeline> = self
            .bindings
            .iter()
            .rfind(|item| Arc::clone(item).downcast::<Pipeline>().is_ok())
            .map(|item| Arc::clone(item).downcast().unwrap())
            .unwrap();

        let layout = pipeline.pipeline_layout.clone();

        self.bindings.push_back(layout.clone());

        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .cmd_bind_descriptor_sets(
                    self.handle,
                    pipeline.bind_point,
                    layout.handle,
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
}

impl CommandBufferBuilder<PipelineBound> {
    pub fn bind_pipeline(self, pipeline: Arc<Pipeline>) -> CommandBufferBuilder<PipelineBound> {
        let new = CommandBufferBuilder {
            handle: self.handle,
            command_buffer_allocator: self.command_buffer_allocator,
            bindings: self.bindings,
            _state: PhantomData,
        };
        new.bind_pipeline_in(pipeline)
    }

    pub fn begin_render_pass(
        self,
        render_target: Arc<RenderTarget>,
        image_index: u32,
    ) -> CommandBufferBuilder<InRenderPassWithPipeline> {
        let new = CommandBufferBuilder {
            handle: self.handle,
            command_buffer_allocator: self.command_buffer_allocator,
            bindings: self.bindings,
            _state: PhantomData,
        };
        new.begin_render_pass_in(render_target, image_index)
    }

    pub fn dispatch(self, group_count: [u32; 3]) -> Self {
        unsafe {
            self.command_buffer_allocator.device.handle.cmd_dispatch(
                self.handle,
                group_count[0],
                group_count[1],
                group_count[2],
            );
        }

        self
    }
}

impl<State: RenderPassBound> CommandBufferBuilder<State> {
    pub fn bind_viewport_and_scissor(
        self,
        viewports: Vec<vk::Viewport>,
        scissors: Vec<vk::Rect2D>,
    ) -> Self {
        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .cmd_set_viewport(self.handle, 0, &viewports);
            self.command_buffer_allocator
                .device
                .handle
                .cmd_set_scissor(self.handle, 0, &scissors);
        }
        self
    }

    pub fn bind_index_buffer(
        mut self,
        buffer: Arc<RwLock<Buffer<IndexBuffer>>>,
        index_type: vk::IndexType,
    ) -> Self {
        let buffer_lock = buffer.read().unwrap();

        self.bindings.push_back(buffer.clone());

        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .cmd_bind_index_buffer(self.handle, buffer_lock.handle, 0, index_type);
        };

        self
    }

    pub fn bind_vertex_buffer(mut self, buffer: Arc<RwLock<Buffer<VertexBuffer>>>) -> Self {
        let buffer_lock = buffer.read().unwrap();
        self.bindings.push_back(buffer.clone());
        let buffer_raw = buffer_lock.handle;

        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .cmd_bind_vertex_buffers(self.handle, 0, &[buffer_raw], &[0]);
        };

        self
    }
}

impl CommandBufferBuilder<InRenderPass> {
    pub fn bind_pipeline(
        self,
        pipeline: Arc<Pipeline>,
    ) -> CommandBufferBuilder<InRenderPassWithPipeline> {
        let new = CommandBufferBuilder {
            handle: self.handle,
            command_buffer_allocator: self.command_buffer_allocator,
            bindings: self.bindings,
            _state: PhantomData,
        };
        new.bind_pipeline_in(pipeline)
    }
}

impl CommandBufferBuilder<InRenderPassWithPipeline> {
    pub fn bind_pipeline(
        self,
        pipeline: Arc<Pipeline>,
    ) -> CommandBufferBuilder<InRenderPassWithPipeline> {
        let new = CommandBufferBuilder {
            handle: self.handle,
            command_buffer_allocator: self.command_buffer_allocator,
            bindings: self.bindings,
            _state: PhantomData,
        };
        new.bind_pipeline_in(pipeline)
    }

    pub fn end_render_pass(self) -> CommandBufferBuilder<PipelineBound> {
        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .cmd_end_render_pass(self.handle)
        };

        CommandBufferBuilder {
            handle: self.handle,
            command_buffer_allocator: self.command_buffer_allocator,
            bindings: self.bindings,
            _state: PhantomData,
        }
    }

    pub fn draw_indexed(
        self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> Self {
        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .cmd_draw_indexed(
                    self.handle,
                    index_count,
                    instance_count,
                    first_index,
                    vertex_offset,
                    first_instance,
                );
        }
        self
    }

    pub fn draw_indexed_inderect(
        self,
        buffer: Arc<RwLock<Buffer<InderectBuffer>>>,
        offset: u64,
        draw_count: u32,
        stride: u32,
    ) -> Self {
        unsafe {
            let lock = buffer.read().unwrap();
            self.command_buffer_allocator
                .device
                .handle
                .cmd_draw_indexed_indirect(self.handle, lock.handle, offset, draw_count, stride);
        }
        self
    }
}

impl CommandBufferBuilder {
    pub fn generate_mipmaps(mut self, image: Arc<Image>) -> CommandBufferBuilder<Idle> {
        self.bindings.push_back(image.clone());
        let mut barrier = vk::ImageMemoryBarrier::default()
            .image(image.handle)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .level_count(1),
            );

        let mut mip_width = image.info.extent[0];
        let mut mip_heigth = image.info.extent[1];

        for mip_level in 1..image.info.mip_levels {
            barrier.subresource_range = barrier.subresource_range.base_mip_level(mip_level - 1);
            barrier = barrier.old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            barrier = barrier.new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            barrier = barrier.src_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            barrier = barrier.dst_access_mask(vk::AccessFlags::TRANSFER_READ);

            unsafe {
                self.command_buffer_allocator
                    .device
                    .handle
                    .cmd_pipeline_barrier(
                        self.handle,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    );
            }

            let blit = vk::ImageBlit::default()
                .src_offsets([
                    vk::Offset3D::default().x(0).y(0).z(0),
                    vk::Offset3D::default()
                        .x(mip_width as i32)
                        .y(mip_heigth as i32)
                        .z(1),
                ])
                .dst_offsets([
                    vk::Offset3D::default().x(0).y(0).z(0),
                    vk::Offset3D::default()
                        .x(if mip_width > 1 {
                            mip_width as i32 / 2
                        } else {
                            1
                        })
                        .y(if mip_heigth > 1 {
                            mip_heigth as i32 / 2
                        } else {
                            1
                        })
                        .z(1),
                ])
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(mip_level - 1)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(mip_level)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            unsafe {
                self.command_buffer_allocator.device.handle.cmd_blit_image(
                    self.handle,
                    image.handle,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit],
                    vk::Filter::LINEAR,
                );
            }

            barrier = barrier.old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            barrier = barrier.new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            barrier = barrier.src_access_mask(vk::AccessFlags::TRANSFER_READ);
            barrier = barrier.dst_access_mask(vk::AccessFlags::SHADER_READ);

            unsafe {
                self.command_buffer_allocator
                    .device
                    .handle
                    .cmd_pipeline_barrier(
                        self.handle,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    );
            }

            if mip_width > 1 {
                mip_width /= 2
            }

            if mip_heigth > 1 {
                mip_heigth /= 2
            }
        }

        barrier.subresource_range = barrier
            .subresource_range
            .base_mip_level(image.info.mip_levels - 1);
        barrier = barrier.old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        barrier = barrier.new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        barrier = barrier.src_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        barrier = barrier.dst_access_mask(vk::AccessFlags::SHADER_READ);

        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .cmd_pipeline_barrier(
                    self.handle,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
        }

        CommandBufferBuilder {
            handle: self.handle,
            command_buffer_allocator: self.command_buffer_allocator,
            bindings: self.bindings,
            _state: PhantomData,
        }
    }
    pub fn transition_image_layout(
        mut self,
        image: Arc<Image>,
        layout_new: vk::ImageLayout,
    ) -> Self {
        self.bindings.push_back(image.clone());

        let layout_old = image.info.layout.get();

        let mut barrier = vk::ImageMemoryBarrier::default()
            .old_layout(layout_old)
            .new_layout(layout_new)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image.handle)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(image.info.mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let mut src_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
        let mut dst_stage = vk::PipelineStageFlags::TRANSFER;

        if layout_old == vk::ImageLayout::UNDEFINED
            && layout_new == vk::ImageLayout::TRANSFER_DST_OPTIMAL
        {
            barrier = barrier
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        } else if layout_old == vk::ImageLayout::TRANSFER_DST_OPTIMAL
            && layout_new == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        {
            barrier = barrier
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);

            src_stage = vk::PipelineStageFlags::TRANSFER;
            dst_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
        }

        unsafe {
            image.info.layout.set(layout_new);

            self.command_buffer_allocator
                .device
                .handle
                .cmd_pipeline_barrier(
                    self.handle,
                    src_stage,
                    dst_stage,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                )
        };

        self
    }

    pub fn copy_image_to_buffer(
        mut self,
        buffer: Arc<RwLock<Buffer<AnyBuffer>>>,
        image: Arc<Image>,
    ) -> Self {
        self = self.transition_image_layout(image.clone(), vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        self.bindings.push_back(image.clone());
        self.bindings.push_back(buffer.clone());

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(image.info.array_layers),
            )
            .image_offset(vk::Offset3D::default())
            .image_extent(vk::Extent3D {
                width: image.info.extent[0],
                height: image.info.extent[1],
                depth: 1,
            });

        let lock = buffer.read().unwrap();

        unsafe {
            lock.device.handle.cmd_copy_image_to_buffer(
                self.handle,
                image.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                lock.handle,
                &[region],
            );
        }

        CommandBufferBuilder {
            handle: self.handle,
            command_buffer_allocator: self.command_buffer_allocator,
            bindings: self.bindings,
            _state: PhantomData,
        }
    }

    pub fn copy_buffer_to_image(
        mut self,
        image: Arc<Image>,
        buffer: Arc<RwLock<Buffer<AnyBuffer>>>,
    ) -> Self {
        self = self.transition_image_layout(image.clone(), vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        self.bindings.push_back(image.clone());
        self.bindings.push_back(buffer.clone());

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(image.info.array_layers),
            )
            .image_offset(vk::Offset3D::default())
            .image_extent(vk::Extent3D {
                width: image.info.extent[0],
                height: image.info.extent[1],
                depth: 1,
            });

        let lock = buffer.read().unwrap();

        unsafe {
            lock.device.handle.cmd_copy_buffer_to_image(
                self.handle,
                lock.handle,
                image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }

        CommandBufferBuilder {
            handle: self.handle,
            command_buffer_allocator: self.command_buffer_allocator,
            bindings: self.bindings,
            _state: PhantomData,
        }
    }

    fn begin_render_pass_in<T: RenderPassBound>(
        mut self,
        render_target: Arc<RenderTarget>,
        image_index: u32,
    ) -> CommandBufferBuilder<T> {
        self.bindings.push_back(render_target.clone());
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

        CommandBufferBuilder {
            handle: self.handle,
            command_buffer_allocator: self.command_buffer_allocator,
            bindings: self.bindings,
            _state: PhantomData,
        }
    }

    fn bind_pipeline_in<OutState: PipelineBoundState>(
        mut self,
        pipeline: Arc<Pipeline>,
    ) -> CommandBufferBuilder<OutState> {
        self.bindings.push_back(pipeline.clone());
        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .cmd_bind_pipeline(self.handle, pipeline.bind_point, pipeline.handle)
        };

        CommandBufferBuilder {
            handle: self.handle,
            command_buffer_allocator: self.command_buffer_allocator,
            bindings: self.bindings,
            _state: PhantomData,
        }
    }
}
