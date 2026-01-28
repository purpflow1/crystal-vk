use std::{
    sync::{Arc, Mutex, RwLock},
    time::SystemTime,
};

use crystal_vk::{
    buffer::Buffer,
    command::CommandBufferAllocator,
    device::{Device, queue::QueuePool},
    image::sampler::Sampler,
    pipeline::{
        Pipeline,
        attribute::{Attribute, AttributeDescriptor},
        descriptor::descriptor_set_layout::descriptor_set::DescriptorSet,
    },
    render::{RenderTarget, swapchain::Swapchain},
    sync::CommandBufferFuture,
};

type Vec3 = [f32; 3];
type Vec2 = [f32; 2];

pub type Index = u16;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VertexTexture(pub Vec3, pub Vec2);

impl AttributeDescriptor for VertexTexture {
    fn get_attributes() -> &'static [Attribute] {
        &[
            Attribute {
                size: size_of::<Vec3>(),
                offset: 0,
            },
            Attribute {
                size: size_of::<Vec2>(),
                offset: size_of::<Vec3>(),
            },
        ]
    }
}

pub struct VulkanContext {
    pub device: Arc<Device>,
    pub queues: QueuePool,
    pub post_process_render_target: Arc<RenderTarget>,
    pub swapchain_render_target: Arc<RenderTarget>,
    pub swapchain: Arc<Swapchain>,

    pub command_allocator: Arc<CommandBufferAllocator>,

    pub buffer_vert: Arc<RwLock<Buffer<VertexTexture>>>,
    pub buffer_ind: Arc<RwLock<Buffer<Index>>>,
    pub buffer_model: Arc<RwLock<Buffer<glam::Mat4>>>,
    pub buffer_resolution_uniform: Arc<RwLock<Buffer<glam::Vec2>>>,
    pub post_process_sampler: Arc<Sampler>,

    pub post_process_pipeline: Arc<Pipeline<VertexTexture>>,
    pub pipeline: Arc<Pipeline<VertexTexture>>,
    pub per_object_descriptor_set: Arc<Mutex<DescriptorSet>>,
    pub post_process_descriptor_set: Arc<Mutex<DescriptorSet>>,

    pub startup_time: SystemTime,
    pub last_frame: SystemTime,

    pub _prev_future: Option<Box<CommandBufferFuture>>,
    pub recreate_swapchain: bool,

    pub extent: [u32; 2],
}
