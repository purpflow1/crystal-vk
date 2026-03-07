use super::*;

use std::{
    collections::BTreeMap,
    ffi::CString,
    fs::File,
    io::{BufReader, Read},
    sync::{Arc, atomic::AtomicBool},
};

use crystal_vk::{
    buffer::{Buffer, BufferInfo},
    command::{CommandBufferAllocator, command_buffer_builder::CommandBufferBuilder},
    device::Device,
    image::sampler::{Sampler, SamplerInfo},
    pipeline::{
        attribute::Attribute,
        descriptor::{
            DescriptorPool,
            descriptor_set_layout::{
                DescriptorSetLayout, LayoutAllocInfo, descriptor_set::DescriptorSet,
            },
            layout::PipelineLayout,
        },
        graphics::GraphicsPipelineInfo,
        shader::Shader,
    },
    render::{RenderTarget, swapchain::Swapchain},
    vk,
};
use winit::{dpi::LogicalSize, window::Window};

use crate::{timeline::Timeline, vulkan_context::VulkanContext};

impl VulkanContext {
    pub fn new(event_loop: &winit::event_loop::ActiveEventLoop) -> (Self, Window) {
        let window = {
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_inner_size(LogicalSize::new(300, 300))
                        .with_min_inner_size(LogicalSize::new(300, 300)),
                )
                .unwrap()
        };

        let instance = crystal_vk::instance::Instance::new_window(&window).unwrap();
        let surface = instance.create_surface().unwrap();
        let physical_device =
            instance.enumerate_physical_devices(Some(surface)).unwrap()[0].clone();

        let (device, queues) = Device::new(
            physical_device,
            vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true),
            vec![
                vk::KHR_SWAPCHAIN_NAME,
                vk::EXT_IMAGE_COMPRESSION_CONTROL_NAME,
                vk::EXT_IMAGE_COMPRESSION_CONTROL_SWAPCHAIN_NAME,
            ],
        )
        .unwrap();

        let present_queue = queues
            .iter()
            .find_map(|(key, val)| if key.present_support { Some(val) } else { None })
            .unwrap()[0]
            .clone();

        let swapchain = Swapchain::new(present_queue, [300, 300], true).unwrap();
        let swapchain_images = swapchain.image_sequence.clone();

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .descriptor_count(3)
                .ty(vk::DescriptorType::UNIFORM_BUFFER),
            vk::DescriptorPoolSize::default()
                .descriptor_count(3)
                .ty(vk::DescriptorType::STORAGE_BUFFER),
            vk::DescriptorPoolSize::default()
                .descriptor_count(3)
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
        ];

        let descriptor_pool = DescriptorPool::new(device.clone(), &pool_sizes).unwrap();

        let mut layout_alloc_infos = BTreeMap::new();
        layout_alloc_infos.insert(
            0,
            LayoutAllocInfo {
                stages: vk::ShaderStageFlags::VERTEX,
                typ: vk::DescriptorType::STORAGE_BUFFER,
                count: 1,
            },
        );
        layout_alloc_infos.insert(
            1,
            LayoutAllocInfo {
                stages: vk::ShaderStageFlags::FRAGMENT,
                typ: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        );

        let per_object_descriptor_set_layout =
            DescriptorSetLayout::new(device.clone(), layout_alloc_infos).unwrap();

        let per_object_descriptor_set = DescriptorSet::new(
            descriptor_pool.clone(),
            per_object_descriptor_set_layout.clone(),
            1,
        )
        .unwrap()[0]
            .clone();

        let per_object_pipeline_layout = PipelineLayout::new(
            descriptor_pool.clone(),
            vec![per_object_descriptor_set_layout],
            &[],
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

        let textured_frag = glsl2spirv!(
            "examples/shaders/textured.frag",
            shaderc::ShaderKind::Fragment
        );

        let entry_point = CString::new("main").unwrap();

        let shader_textured_vert = Shader::new(
            device.clone(),
            entry_point.clone(),
            vk::ShaderStageFlags::VERTEX,
            textured_vert.as_binary().to_vec(),
        )
        .unwrap();

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
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        )
        .unwrap();

        let post_process_render_target =
            RenderTarget::new(device.clone(), vec![post_process_stage_image.clone()], 4).unwrap();
        let swapchain_render_target =
            RenderTarget::new(device.clone(), swapchain_images, 4).unwrap();

        let world_object_pipeline = crystal_vk::pipeline::Pipeline::new_graphics(
            per_object_pipeline_layout,
            post_process_render_target.clone(),
            vec![shader_textured_vert, shader_textured_frag],
            GraphicsPipelineInfo {
                vertex_attributes: vec![
                    Attribute {
                        size: size_of::<[f32; 3]>(),
                        offset: 0,
                    },
                    Attribute {
                        size: size_of::<[f32; 2]>(),
                        offset: size_of::<[f32; 3]>(),
                    },
                ],
                ..Default::default()
            },
            None,
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

        let buffer_vertex = Buffer::new(
            device.clone(),
            BufferInfo {
                size: 12 * size_of::<VertexTexture>() as u64,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                usage: vk::BufferUsageFlags::VERTEX_BUFFER,
                properties: vk::MemoryPropertyFlags::HOST_VISIBLE,
            },
        )
        .unwrap();

        {
            let mut lock = buffer_vertex.write().unwrap();
            let size = lock.info.size;
            let memory = lock.bind_memory(0..size).unwrap();
            memory.copy_from_slice(bytemuck::cast_slice(&[
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
            ]));
        }

        let buffer_index = Buffer::new(
            device.clone(),
            BufferInfo {
                size: 42 * size_of::<Index>() as u64,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                usage: vk::BufferUsageFlags::INDEX_BUFFER,
                properties: vk::MemoryPropertyFlags::HOST_VISIBLE,
            },
        )
        .unwrap();

        {
            let mut lock = buffer_index.write().unwrap();
            let size = lock.info.size;
            let memory = lock.bind_memory(0..size).unwrap();
            memory.copy_from_slice(bytemuck::cast_slice::<u16, u8>(&[
                0, 2, 1, 1, 2, 3, // bottom
                4, 5, 6, 5, 7, 6, // top
                0, 4, 2, 2, 4, 6, // front
                1, 3, 5, 3, 7, 5, // back
                0, 1, 4, 1, 5, 4, // right
                2, 6, 3, 3, 6, 7, // left
                0, 2, 1, 3, 2, 0, // screen plane
            ]));
        }

        let buffer_model = Buffer::new(
            device.clone(),
            BufferInfo {
                size: size_of::<glam::Mat4>() as u64,
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
            let mut decoder = png::Decoder::new(buf_reader);
            decoder.set_transformations(png::Transformations::all());
            let mut reader = decoder.read_info().unwrap();
            let size = reader.output_buffer_size().unwrap();

            let buffer = Buffer::new(
                device.clone(),
                BufferInfo {
                    size: (size * 2) as u64,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC,
                    properties: vk::MemoryPropertyFlags::HOST_COHERENT
                        | vk::MemoryPropertyFlags::HOST_VISIBLE,
                },
            )
            .unwrap();

            let mut lock = buffer.write().unwrap();
            let memory = lock.bind_memory(0..size as u64).unwrap();
            let info = reader.next_frame(memory).unwrap();
            drop(lock);

            (
                crystal_vk::image::Image::new(
                    device.clone(),
                    [info.width, info.height],
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::COLOR_ATTACHMENT,
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

        let mut future =
            CommandBufferBuilder::new(command_allocator.clone(), transfer_queue_family_info.index)
                .unwrap()
                .copy_buffer_to_image(image.clone(), image_buffer.clone())
                .generate_mipmaps(image.clone())
                .build(transfer_queue)
                .unwrap();

        future.flush().unwrap();
        future.wait().unwrap();

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
            1,
        )
        .unwrap()[0]
            .clone();

        let buffer_resolution_uniform = Buffer::new(
            device.clone(),
            BufferInfo {
                size: size_of::<glam::Vec2>() as u64,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                properties: vk::MemoryPropertyFlags::HOST_VISIBLE,
            },
        )
        .unwrap();

        let mut lock = buffer_resolution_uniform.write().unwrap();
        let memory = lock.bind_memory(0..8).unwrap();
        memory.copy_from_slice(bytemuck::cast_slice(&[
            window.inner_size().width as f32,
            window.inner_size().height as f32,
        ]));

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
            &[],
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

        let post_process_pipeline = crystal_vk::pipeline::Pipeline::new_graphics(
            post_process_pipeline_layout,
            swapchain_render_target.clone(),
            vec![post_process_vert, post_process_frag],
            GraphicsPipelineInfo {
                vertex_attributes: vec![
                    Attribute {
                        size: size_of::<[f32; 3]>(),
                        offset: 0,
                    },
                    Attribute {
                        size: size_of::<[f32; 2]>(),
                        offset: size_of::<[f32; 3]>(),
                    },
                ],
                ..Default::default()
            },
            None,
        )
        .unwrap();

        let mut lock = per_object_descriptor_set.lock().unwrap();
        lock.bind_buffer(buffer_model.clone(), 0, 0, 1).unwrap();
        lock.bind_combined_image_sampler(image.clone(), sampler.clone(), 1, 0, 1)
            .unwrap();

        drop(lock);

        (
            Self {
                heartbeat: Arc::new(AtomicBool::new(true)),
                stop_flag: Arc::new(AtomicBool::new(false)),
                watcher: None,
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

                timeline: Timeline::new(),
                first_frame: true,

                prev_future: None,

                extent: [1200, 800],
                extent_changed: false,
            },
            window,
        )
    }
}
