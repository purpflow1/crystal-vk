use std::{
    collections::BTreeMap,
    f32::consts::PI,
    ffi::CString,
    fs::File,
    io::{BufReader, Read},
    sync::{Arc, Mutex, RwLock},
    time::SystemTime,
};

use ash::vk::{self, DescriptorType, ShaderStageFlags};
use crystal_vk::{
    buffer::{Buffer, BufferCreateInfo},
    command::{CommandBufferAllocator, command_buffer_builder::CommandBufferBuilder},
    device::{
        Device,
        queue::{Queue, QueuePool},
    },
    image::sampler::{Sampler, SamplerInfo},
    pipeline::{
        Pipeline, PipelineInfo,
        attribute::{Attribute, AttributeDescriptor},
        descriptor::{
            DescriptorPool,
            descriptor_set_layout::{
                DescriptorSetLayout, LayoutAllocInfo, descriptor_set::DescriptorSet,
            },
            layout::PipelineLayout,
        },
        shader::Shader,
    },
    render::{RenderTarget, swapchain::Swapchain},
    sync::{CommandBufferFuture, GpuFuture, PresentFuture, SwapchainFuture},
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

type Vec3 = [f32; 3];
type Vec2 = [f32; 2];

pub type Index = u16;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VertexTexture(Vec3, Vec2);

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

#[derive(Default)]
struct ContextWindow {
    data: Option<Data>,
    window: Option<Window>,
}

struct Data {
    device: Arc<Device>,
    queues: QueuePool,
    post_process_render_target: Arc<RenderTarget>,
    swapchain_render_target: Arc<RenderTarget>,
    swapchain: Arc<Swapchain>,

    command_allocator: Arc<CommandBufferAllocator>,

    buffer_vert: Arc<RwLock<Buffer<VertexTexture>>>,
    buffer_ind: Arc<RwLock<Buffer<Index>>>,
    buffer_model: Arc<RwLock<Buffer<glam::Mat4>>>,
    buffer_resolution_uniform: Arc<RwLock<Buffer<glam::Vec2>>>,
    post_process_sampler: Arc<Sampler>,

    post_process_pipeline: Arc<Pipeline<VertexTexture>>,
    pipeline: Arc<Pipeline<VertexTexture>>,
    per_object_descriptor_set: Arc<Mutex<DescriptorSet>>,
    post_process_descriptor_set: Arc<Mutex<DescriptorSet>>,

    startup_time: SystemTime,
    last_frame: SystemTime,

    prev_future: Option<Box<CommandBufferFuture>>,
    recreate_swapchain: bool,

    extent: [u32; 2],
}

impl ContextWindow {}

impl ApplicationHandler for ContextWindow {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = {
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_inner_size(LogicalSize::new(300, 300))
                        .with_min_inner_size(LogicalSize::new(100, 100)),
                )
                .unwrap()
        };

        let device = Device::with_present(&window).unwrap();
        let queues = Queue::instantiate(device.clone());

        let present_queue = queues
            .iter()
            .find_map(|(key, val)| if key.present_support { Some(val) } else { None })
            .unwrap()[0]
            .clone();

        let swapchain = Swapchain::new(present_queue, [600, 600], true).unwrap();
        let swapchain_images = swapchain.image_sequence.clone();

        let descriptor_pool = DescriptorPool::new(device.clone()).unwrap();

        let mut layout_alloc_infos = BTreeMap::new();
        layout_alloc_infos.insert(
            0,
            LayoutAllocInfo {
                stages: ShaderStageFlags::VERTEX,
                typ: DescriptorType::STORAGE_BUFFER,
                count: 1,
            },
        );
        layout_alloc_infos.insert(
            1,
            LayoutAllocInfo {
                stages: ShaderStageFlags::FRAGMENT,
                typ: DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        );

        let per_object_descriptor_set_layout =
            DescriptorSetLayout::new(device.clone(), layout_alloc_infos).unwrap();

        let per_object_descriptor_set = DescriptorSet::new(
            descriptor_pool.clone(),
            per_object_descriptor_set_layout.clone(),
        )
        .unwrap();

        let per_object_pipeline_layout = PipelineLayout::new(
            descriptor_pool.clone(),
            vec![per_object_descriptor_set_layout],
        )
        .unwrap();

        let compiler = shaderc::Compiler::new().unwrap();

        macro_rules! glsl2spirv {
            ($filename:expr, $shaderkind:expr) => {{
                const FILENAME: &str = $filename;
                let mut source = String::new();
                let mut reader = BufReader::new(File::open(FILENAME).unwrap());
                reader.read_to_string(&mut source).unwrap();
                compiler
                    .compile_into_spirv(source.as_str(), $shaderkind, FILENAME, "main", None)
                    .unwrap()
            }};
        }

        let textured_vert = glsl2spirv!(
            "examples/shaders/textured.vert",
            shaderc::ShaderKind::Vertex
        );

        let entry_point = CString::new("main").unwrap();

        let shader_textured_vert = Shader::new(
            device.clone(),
            entry_point.clone(),
            vk::ShaderStageFlags::VERTEX,
            textured_vert.as_binary().to_vec(),
        )
        .unwrap();

        let textured_frag = glsl2spirv!(
            "examples/shaders/textured.frag",
            shaderc::ShaderKind::Fragment
        );

        let shader_textured_frag = Shader::new(
            device.clone(),
            entry_point.clone(),
            vk::ShaderStageFlags::FRAGMENT,
            textured_frag.as_binary().to_vec(),
        )
        .unwrap();

        let post_process_stage_image = crystal_vk::image::Image::new(
            device.clone(),
            [window.inner_size().width, window.inner_size().height],
            vk::Format::R8G8B8A8_SRGB,
        )
        .unwrap();

        let post_process_render_target =
            RenderTarget::new(device.clone(), vec![post_process_stage_image.clone()], 4).unwrap();
        let swapchain_render_target =
            RenderTarget::new(device.clone(), swapchain_images, 4).unwrap();

        let world_object_pipeline = crystal_vk::pipeline::Pipeline::<VertexTexture>::new_graphics(
            per_object_pipeline_layout,
            post_process_render_target.clone(),
            vec![shader_textured_vert, shader_textured_frag],
            PipelineInfo::default(),
        )
        .unwrap();

        let post_process_vert = glsl2spirv!(
            "examples/shaders/post-process.vert",
            shaderc::ShaderKind::Vertex
        );
        let post_process_frag = glsl2spirv!(
            "examples/shaders/post-process.frag",
            shaderc::ShaderKind::Fragment
        );

        let buffer_vertex = Buffer::<VertexTexture>::new(
            device.clone(),
            BufferCreateInfo {
                len: 12,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                usage: vk::BufferUsageFlags::VERTEX_BUFFER,
                properties: vk::MemoryPropertyFlags::HOST_VISIBLE,
            },
        )
        .unwrap();

        {
            let mut lock = buffer_vertex.write().unwrap();
            let buffer = &mut lock[..];
            buffer.copy_from_slice(&[
                // cube bottom
                VertexTexture([0.5, -0.5, 0.5], [0.0, 0.0]),
                VertexTexture([0.5, -0.5, -0.5], [1.0, 0.0]),
                VertexTexture([-0.5, -0.5, 0.5], [0.0, 1.0]),
                VertexTexture([-0.5, -0.5, -0.5], [1.0, 1.0]),
                // cube top
                VertexTexture([0.5, 0.5, 0.5], [0.0, 0.0]),
                VertexTexture([0.5, 0.5, -0.5], [1.0, 0.0]),
                VertexTexture([-0.5, 0.5, 0.5], [0.0, 1.0]),
                VertexTexture([-0.5, 0.5, -0.5], [1.0, 1.0]),
                // screen plane
                VertexTexture([-1., -1., 0.], [0., 0.]),
                VertexTexture([1., -1., 0.], [1., 0.]),
                VertexTexture([1., 1., 0.], [1., 1.]),
                VertexTexture([-1., 1., 0.], [0., 1.]),
            ]);
        }

        let buffer_index = Buffer::<Index>::new(
            device.clone(),
            BufferCreateInfo {
                len: 42,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                usage: vk::BufferUsageFlags::INDEX_BUFFER,
                properties: vk::MemoryPropertyFlags::HOST_VISIBLE,
            },
        )
        .unwrap();

        {
            let mut lock = buffer_index.write().unwrap();
            let buffer = &mut lock[..];
            buffer.copy_from_slice(&[
                0, 2, 1, 1, 2, 3, // bottom
                4, 5, 6, 5, 7, 6, // top
                0, 4, 2, 2, 4, 6, // front
                1, 3, 5, 3, 7, 5, // back
                0, 1, 4, 1, 5, 4, // right
                2, 6, 3, 3, 6, 7, // left
                0, 2, 1, 3, 2, 0, // screen plane
            ]);
        }

        let buffer_model = Buffer::<glam::Mat4>::new(
            device.clone(),
            BufferCreateInfo {
                len: 1,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                properties: vk::MemoryPropertyFlags::HOST_VISIBLE,
            },
        )
        .unwrap();

        let sampler = Sampler::new(
            device.clone(),
            SamplerInfo {
                filter: vk::Filter::LINEAR,
                address_mode: vk::SamplerAddressMode::REPEAT,
                anisotropy_texels: 1.,
                max_lod: 0.,
            },
        )
        .unwrap();

        let command_allocator = CommandBufferAllocator::new(queues.clone()).unwrap();

        let (image, image_buffer) = {
            let file = File::open("examples/resources/textures/test.png").unwrap();
            let buf_reader = BufReader::new(file);
            let decoder = png::Decoder::new(buf_reader);
            let mut reader = decoder.read_info().unwrap();
            let size = reader.output_buffer_size().unwrap();

            let buffer = Buffer::new(
                device.clone(),
                BufferCreateInfo {
                    len: size as u64,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC,
                    properties: vk::MemoryPropertyFlags::HOST_COHERENT
                        | vk::MemoryPropertyFlags::HOST_VISIBLE,
                },
            )
            .unwrap();

            let mut lock = buffer.write().unwrap();
            let info = reader.next_frame(&mut lock[..size as u64]).unwrap();
            drop(lock);

            (
                crystal_vk::image::Image::new(
                    device.clone(),
                    [info.width, info.height],
                    vk::Format::R8G8B8A8_UNORM,
                )
                .unwrap(),
                buffer,
            )
        };

        let (transfer_queue_family_info, transfer_queue) = queues
            .iter()
            .find_map(|(info, queues)| {
                if info.flags.contains(vk::QueueFlags::TRANSFER) {
                    Some((*info, queues[0].clone()))
                } else {
                    None
                }
            })
            .unwrap();

        let command_buffer_builder = CommandBufferBuilder::new(
            command_allocator.clone(),
            transfer_queue_family_info.index,
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        )
        .unwrap()
        .transition_image_layout(image.clone(), vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .unwrap()
        .stage_image(image.clone(), image_buffer.clone())
        .unwrap()
        .generate_mipmaps(image.clone())
        .unwrap()
        .build()
        .unwrap();

        let mut command_buffer = command_buffer_builder.execute(transfer_queue).unwrap();
        command_buffer.flush().unwrap();
        command_buffer.wait().unwrap();

        let mut layout_alloc_info = BTreeMap::new();

        layout_alloc_info.insert(
            0,
            LayoutAllocInfo {
                stages: vk::ShaderStageFlags::FRAGMENT,
                typ: vk::DescriptorType::UNIFORM_BUFFER,
                count: 1,
            },
        );

        layout_alloc_info.insert(
            1,
            LayoutAllocInfo {
                stages: vk::ShaderStageFlags::FRAGMENT,
                typ: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        );

        let post_process_descriptor_set_layout =
            DescriptorSetLayout::new(device.clone(), layout_alloc_info).unwrap();

        let post_process_descriptor_set = DescriptorSet::new(
            descriptor_pool.clone(),
            post_process_descriptor_set_layout.clone(),
        )
        .unwrap();

        let buffer_resolution_uniform = Buffer::new(
            device.clone(),
            BufferCreateInfo {
                len: 1,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                properties: vk::MemoryPropertyFlags::HOST_VISIBLE,
            },
        )
        .unwrap();

        let mut lock = buffer_resolution_uniform.write().unwrap();

        lock[0] = glam::Vec2::new(
            window.inner_size().width as f32,
            window.inner_size().height as f32,
        );

        drop(lock);

        let post_process_sampler = Sampler::new(
            device.clone(),
            SamplerInfo {
                filter: vk::Filter::NEAREST,
                address_mode: vk::SamplerAddressMode::REPEAT,
                anisotropy_texels: 1.,
                max_lod: 0.,
            },
        )
        .unwrap();

        let mut lock = post_process_descriptor_set.lock().unwrap();

        lock.bind_buffer(buffer_resolution_uniform.clone(), 0, 0, 1)
            .unwrap();
        lock.bind_combined_image_sampler(
            post_process_stage_image,
            post_process_sampler.clone(),
            1,
            0,
            0,
        )
        .unwrap();

        drop(lock);

        let post_process_pipeline_layout = PipelineLayout::new(
            descriptor_pool.clone(),
            vec![post_process_descriptor_set_layout],
        )
        .unwrap();

        let post_process_vert = Shader::new(
            device.clone(),
            CString::new("main").unwrap(),
            vk::ShaderStageFlags::VERTEX,
            post_process_vert.as_binary().to_vec(),
        )
        .unwrap();

        let post_process_frag = Shader::new(
            device.clone(),
            CString::new("main").unwrap(),
            vk::ShaderStageFlags::FRAGMENT,
            post_process_frag.as_binary().to_vec(),
        )
        .unwrap();

        let post_process_pipeline = crystal_vk::pipeline::Pipeline::<VertexTexture>::new_graphics(
            post_process_pipeline_layout,
            swapchain_render_target.clone(),
            vec![post_process_vert, post_process_frag],
            PipelineInfo::default(),
        )
        .unwrap();

        let mut lock = per_object_descriptor_set.lock().unwrap();
        lock.bind_buffer(buffer_model.clone(), 0, 0, 1).unwrap();
        lock.bind_combined_image_sampler(image.clone(), sampler.clone(), 1, 0, 1)
            .unwrap();

        drop(lock);

        self.window = Some(window);
        self.data = Some(Data {
            device,
            queues,
            post_process_render_target,
            swapchain_render_target,
            swapchain,

            command_allocator,

            buffer_ind: buffer_index,
            buffer_model,
            buffer_vert: buffer_vertex,
            buffer_resolution_uniform,
            post_process_sampler,

            post_process_pipeline,
            pipeline: world_object_pipeline,
            per_object_descriptor_set: per_object_descriptor_set.clone(),
            post_process_descriptor_set,

            startup_time: SystemTime::now(),
            last_frame: SystemTime::UNIX_EPOCH,

            prev_future: None,
            recreate_swapchain: false,

            extent: [1200, 800],
        });
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {}

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let window = self.window.as_ref().unwrap();

        let data = self.data.as_mut().unwrap();

        let aspect_ratio = data.extent[0] as f32 / data.extent[1] as f32;

        let delta_time = SystemTime::now().duration_since(data.last_frame).unwrap();

        window.set_title(format!("FPS: {}", (1. / delta_time.as_secs_f32()) as u32).as_str());

        let camera = glam::Mat4::look_at_lh(
            glam::Vec3::new(0., 0., -1.),
            glam::Vec3::ZERO,
            glam::Vec3::new(0., 1., 0.),
        );

        let perspective = glam::Mat4::perspective_lh(PI / 3., aspect_ratio, 0.1, 100.);

        let render_camera = perspective * camera;

        let mut buffer = data.buffer_model.write().unwrap();

        let seconds = data.startup_time.elapsed().unwrap().as_secs_f32();

        let model = glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::new(0.8, 0.8, 0.8),
            glam::Quat::from_rotation_y(seconds) * glam::Quat::from_rotation_z(seconds),
            glam::Vec3::new(0., 0., 1.),
        );

        buffer[0] = render_camera * model;

        let mut buffer = data.buffer_resolution_uniform.write().unwrap();
        buffer[0] = glam::Vec2::new(data.extent[0] as f32, data.extent[1] as f32);

        drop(buffer);

        let (family_info, queues) = data
            .queues
            .iter()
            .find(|(family, _)| family.flags.contains(vk::QueueFlags::GRAPHICS))
            .unwrap()
            .clone();

        if data.recreate_swapchain {
            data.swapchain = Swapchain::from_old(data.swapchain.clone(), data.extent).unwrap();

            let post_process_image = crystal_vk::image::Image::new(
                data.device.clone(),
                data.extent,
                vk::Format::R8G8B8A8_SRGB,
            )
            .unwrap();

            let mut lock = data.post_process_descriptor_set.lock().unwrap();
            lock.bind_combined_image_sampler(
                post_process_image.clone(),
                data.post_process_sampler.clone(),
                1,
                0,
                1,
            )
            .unwrap();

            drop(lock);

            data.post_process_render_target =
                RenderTarget::new(data.device.clone(), vec![post_process_image.clone()], 4)
                    .unwrap();

            data.swapchain_render_target = RenderTarget::new(
                data.device.clone(),
                data.swapchain.image_sequence.clone(),
                4,
            )
            .unwrap();

            data.recreate_swapchain = false
        }

        // TODO not safe
        let mut swapchain_future =
            SwapchainFuture::new(data.device.clone(), data.swapchain.clone()).unwrap();

        // blocks until aviability
        let (image_index, _suboptimal) = match swapchain_future.acquire_next_image() {
            Ok(result) => result,
            Err(_e) => {
                dbg!(_e);
                return;
            }
        };

        let queue = queues[0].clone();

        let builder = CommandBufferBuilder::new(
            data.command_allocator.clone(),
            family_info.index,
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        )
        .unwrap()
        .begin_render_pass(data.post_process_render_target.clone(), 0)
        .unwrap()
        .bind_viewport_and_scissor(
            vec![vk::Viewport {
                width: data.extent[0] as f32,
                height: data.extent[1] as f32,
                ..Default::default()
            }],
            vec![vk::Rect2D {
                extent: vk::Extent2D {
                    width: data.extent[0],
                    height: data.extent[1],
                },
                ..Default::default()
            }],
        )
        .bind_pipeline(data.pipeline.clone(), vk::PipelineBindPoint::GRAPHICS)
        .bind_vertex_buffer(data.buffer_vert.clone())
        .bind_index_buffer(data.buffer_ind.clone())
        .bind_descriptor_sets(
            data.pipeline.clone(),
            0,
            vec![data.per_object_descriptor_set.clone()],
        )
        .draw_indexed(36, 1, 0, 0, 0)
        .end_render_pass()
        .begin_render_pass(data.swapchain_render_target.clone(), image_index)
        .unwrap()
        .bind_pipeline(
            data.post_process_pipeline.clone(),
            vk::PipelineBindPoint::GRAPHICS,
        )
        .bind_vertex_buffer(data.buffer_vert.clone())
        .bind_index_buffer(data.buffer_ind.clone())
        .bind_descriptor_sets(
            data.post_process_pipeline.clone(),
            0,
            vec![data.post_process_descriptor_set.clone()],
        )
        .draw_indexed(6, 1, 36, 8, 0)
        .end_render_pass();

        let command_buffer = builder.build().unwrap();

        let mut command_buffer_future = command_buffer.execute(queue).unwrap();
        let mut present_future =
            PresentFuture::new(data.device.clone(), data.swapchain.clone()).unwrap();
        command_buffer_future.sync_with_present(&mut present_future);

        command_buffer_future.flush().unwrap();
        data.recreate_swapchain = present_future.present(image_index).unwrap();

        data.last_frame = SystemTime::now();

        match event {
            WindowEvent::CloseRequested => {
                println!("Stopping window context with close request");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => data.extent = [size.width, size.height],
            _ => {
                self.window.as_ref().unwrap().request_redraw();
            }
        }
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        // Wayland surface can be destroyed before vulkan resources removal
        self.data = None;
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut context = ContextWindow::default();
    event_loop
        .run_app(&mut context)
        .expect("cannot run event loop");
}
