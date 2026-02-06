use std::{
    error::Error,
    sync::{Arc, Mutex, RwLock},
};

use ash::vk;

use crate::{
    buffer::Buffer,
    command::CommandBufferAllocator,
    device::queue::Queue,
    image::Image,
    pipeline::{Pipeline, descriptor::descriptor_set_layout::descriptor_set::DescriptorSet},
    render::RenderTarget,
    sync::CommandBufferFuture,
    traits::CommandBufferBinding,
};

/// # Safety
/// Everything is boxed
pub struct CommandBufferBuilder {
    pub(crate) handle: vk::CommandBuffer,
    pub(crate) command_buffer_allocator: Arc<CommandBufferAllocator>,

    bindings: Vec<Arc<dyn CommandBufferBinding>>,
    last_pipeline_bound: Option<Arc<Pipeline>>,
    render_pass_begun: bool,
}

impl CommandBufferBuilder {
    pub fn build(
        self: Box<Self>,
        queue: Arc<Mutex<Queue>>,
    ) -> Result<Box<CommandBufferFuture>, Box<dyn Error>> {
        if self.render_pass_begun {
            self.end_render_pass();
        }

        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .end_command_buffer(self.handle)
        }?;

        CommandBufferFuture::new(self, queue)
    }

    pub fn new(
        command_buffer_allocator: Arc<CommandBufferAllocator>,
        queue_family_index: u32,
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

        Ok(Box::new(Self {
            handle: command_buffer,
            command_buffer_allocator,
            bindings: Vec::new(),
            last_pipeline_bound: None,
            render_pass_begun: false,
        }))
    }

    pub fn transition_image_layout(
        mut self: Box<Self>,
        image: Arc<Image>,
        layout_new: vk::ImageLayout,
    ) -> Result<Box<Self>, Box<dyn Error>> {
        self.bindings.push(image.clone());

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
        } else {
            return Err(
                format!("unsupported layout transition: {layout_old:?} -> {layout_new:?}").into(),
            );
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

        Ok(self)
    }

    pub fn generate_mipmaps(
        mut self: Box<Self>,
        image: Arc<Image>,
    ) -> Result<Box<Self>, Box<dyn Error>> {
        self.bindings.push(image.clone());
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

        Ok(self)
    }

    pub fn stage_image(
        mut self: Box<Self>,
        image: Arc<Image>,
        buffer: Arc<RwLock<Buffer>>,
    ) -> Result<Box<Self>, Box<dyn Error>> {
        self.bindings.push(image.clone());
        self.bindings.push(buffer.clone());
        let layout = image.info.layout.get();

        if layout != vk::ImageLayout::TRANSFER_DST_OPTIMAL {
            return Err(format!("staging is not supported with {layout:?}").into());
        }

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
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

        Ok(self)
    }

    pub fn dispatch(self: Box<Self>, group_count: [u32; 3]) -> Result<Box<Self>, Box<dyn Error>> {
        if self.last_pipeline_bound.is_none() {
            return Err("Pipeline should be bound before the dispatch call!".into());
        }

        unsafe {
            self.command_buffer_allocator.device.handle.cmd_dispatch(
                self.handle,
                group_count[0],
                group_count[1],
                group_count[2],
            );
        }

        Ok(self)
    }

    pub fn draw_indexed(
        self: Box<Self>,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> Result<Box<Self>, Box<dyn Error>> {
        if self.last_pipeline_bound.is_none() {
            return Err("Pipeline should be bound before the draw call!".into());
        }

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
        Ok(self)
    }

    pub fn draw_indexed_inderect(
        self: Box<Self>,
        buffer: Arc<RwLock<Buffer>>,
        offset: u64,
        draw_count: u32,
        stride: u32,
    ) -> Result<Box<Self>, Box<dyn Error>> {
        if self.last_pipeline_bound.is_none() {
            return Err("Pipeline should be bound before the draw call!".into());
        }

        unsafe {
            let lock = buffer.read().unwrap();
            self.command_buffer_allocator
                .device
                .handle
                .cmd_draw_indexed_indirect(self.handle, lock.handle, offset, draw_count, stride);
        }
        Ok(self)
    }

    pub fn bind_descriptor_sets(
        mut self: Box<Self>,
        first_set: u32,
        descriptor_sets: Vec<Arc<Mutex<DescriptorSet>>>,
    ) -> Result<Box<Self>, Box<dyn Error>> {
        for descriptor_set in descriptor_sets.iter() {
            self.bindings.push(descriptor_set.clone());
        }

        let pipeline = if let Some(pipeline) = self.last_pipeline_bound.clone() {
            pipeline
        } else {
            return Err(
                "Pipeline was not bound! Pipeline should be bound before descriptor sets".into(),
            );
        };

        let layout = pipeline.pipeline_layout.clone();

        self.bindings.push(layout.clone());

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
        Ok(self)
    }

    pub fn bind_index_buffer(
        mut self: Box<Self>,
        buffer: Arc<RwLock<Buffer>>,
    ) -> Result<Box<Self>, Box<dyn Error>> {
        let buffer_lock = buffer.read().unwrap();

        if !buffer_lock
            .info
            .usage
            .intersects(vk::BufferUsageFlags::INDEX_BUFFER)
        {
            return Err("Cannot bind index buffer without INDEX_BUFFER usage flags!".into());
        }

        self.bindings.push(buffer.clone());

        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .cmd_bind_index_buffer(self.handle, buffer_lock.handle, 0, vk::IndexType::UINT16);
        };

        Ok(self)
    }

    pub fn bind_vertex_buffer(
        mut self: Box<Self>,
        buffer: Arc<RwLock<Buffer>>,
    ) -> Result<Box<Self>, Box<dyn Error>> {
        let buffer_lock = buffer.read().unwrap();

        if !buffer_lock
            .info
            .usage
            .intersects(vk::BufferUsageFlags::VERTEX_BUFFER)
        {
            return Err("Cannot bind vertex buffer without VERTEX_BUFFER usage flags!".into());
        }

        self.bindings.push(buffer.clone());

        let buffer_raw = buffer_lock.handle;

        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .cmd_bind_vertex_buffers(self.handle, 0, &[buffer_raw], &[0]);
        };

        Ok(self)
    }

    pub fn bind_viewport_and_scissor(
        self: Box<Self>,
        viewports: Vec<vk::Viewport>,
        scissors: Vec<vk::Rect2D>,
    ) -> Box<Self> {
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

    pub fn bind_pipeline(mut self: Box<Self>, pipeline: Arc<Pipeline>) -> Box<Self> {
        self.last_pipeline_bound = Some(pipeline.clone());

        self.bindings.push(pipeline.clone());
        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .cmd_bind_pipeline(self.handle, pipeline.bind_point, pipeline.handle)
        };

        self
    }

    pub fn bind_render_target(
        mut self: Box<Self>,
        render_target: Arc<RenderTarget>,
        image_index: u32,
    ) -> Result<Box<Self>, Box<dyn Error>> {
        if self.render_pass_begun {
            self.end_render_pass();
        }

        self.render_pass_begun = true;

        self.bindings.push(render_target.clone());
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

    fn end_render_pass(&self) {
        unsafe {
            self.command_buffer_allocator
                .device
                .handle
                .cmd_end_render_pass(self.handle)
        };
    }
}
