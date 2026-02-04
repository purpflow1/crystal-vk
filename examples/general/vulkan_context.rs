use std::{
    error::Error,
    pin::Pin,
    sync::{Arc, Mutex, RwLock, atomic::AtomicBool},
};

use crystal_vk::{
    buffer::Buffer,
    command::CommandBufferAllocator,
    device::{Device, queue::QueuePool},
    image::sampler::Sampler,
    pipeline::{Pipeline, descriptor::descriptor_set_layout::descriptor_set::DescriptorSet},
    render::{RenderTarget, swapchain::Swapchain},
};

use crate::timeline;

type Vec3 = [f32; 3];
type Vec2 = [f32; 2];

pub type Index = u16;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VertexTexture(pub Vec3, pub Vec2);

unsafe impl bytemuck::NoUninit for VertexTexture {}

type FutureType = dyn Future<Output = Result<bool, Box<dyn Error>>> + Send + Sync;

pub struct VulkanContext {
    pub heartbeat: Arc<AtomicBool>,
    pub stop_flag: Arc<AtomicBool>,
    pub watcher: Option<std::thread::JoinHandle<()>>,

    pub device: Arc<Device>,
    pub queues: QueuePool,
    pub post_process_render_target: Arc<RenderTarget>,
    pub swapchain_render_target: Arc<RenderTarget>,
    pub swapchain: Arc<Swapchain>,

    pub command_allocator: Arc<CommandBufferAllocator>,

    pub buffer_vert: Arc<RwLock<Buffer>>,
    pub buffer_ind: Arc<RwLock<Buffer>>,
    pub buffer_model: Arc<RwLock<Buffer>>,
    pub buffer_resolution_uniform: Arc<RwLock<Buffer>>,
    pub post_process_sampler: Arc<Sampler>,

    pub post_process_pipeline: Arc<Pipeline>,
    pub pipeline: Arc<Pipeline>,
    pub per_object_descriptor_set: Arc<Mutex<DescriptorSet>>,
    pub post_process_descriptor_set: Arc<Mutex<DescriptorSet>>,

    pub prev_future: Option<Pin<Box<FutureType>>>,

    pub timeline: timeline::Timeline,
    pub first_frame: bool,

    pub extent: [u32; 2],
}
