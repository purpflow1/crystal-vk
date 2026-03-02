use std::{
    error::Error,
    ffi::CString,
    fs::File,
    io::{BufReader, BufWriter, Read},
    time::SystemTime,
};

use ash::vk;
use crystal_vk::{
    buffer::{AnyBuffer, BufferInfo},
    command::{CommandBufferAllocator, command_buffer_builder::CommandBufferBuilder},
    pipeline::{
        Pipeline,
        descriptor::{
            DescriptorPool,
            descriptor_set_layout::{
                DescriptorSetLayout, LayoutAllocInfo, descriptor_set::DescriptorSet,
            },
            layout::PipelineLayout,
        },
        shader::Shader,
    },
};

const SAMPLES: i32 = 1024;
const FRAMES: u32 = 128;

fn main() -> Result<(), Box<dyn Error>> {
    let width = 3000;
    let height = 3000;

    let instance = crystal_vk::instance::Instance::new()?;
    let physical_device = instance.enumerate_physical_devices(None)?[0].clone();
    let (device, queues) = crystal_vk::device::Device::new(
        physical_device,
        vk::PhysicalDeviceFeatures::default(),
        vec![vk::KHR_DEFERRED_HOST_OPERATIONS_NAME],
    )?;

    let queue = queues.first_key_value().unwrap().1[0].clone();

    let allocator = CommandBufferAllocator::new(queues.clone())?;

    let image = crystal_vk::image::Image::new(
        device.clone(),
        [width, height],
        vk::Format::R32G32B32A32_SFLOAT,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
    )?;

    let mut command_buffer = CommandBufferBuilder::new(allocator.clone(), 0)?
        .transition_image_layout(image.clone(), vk::ImageLayout::GENERAL)
        .build(queue.clone())?;

    command_buffer.flush()?;
    command_buffer.wait()?;

    let descriptor_pool = DescriptorPool::new(
        device.clone(),
        &[vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)],
    )?;

    let descriptor_set_layout = DescriptorSetLayout::new(
        device.clone(),
        vec![(
            0,
            LayoutAllocInfo {
                stages: vk::ShaderStageFlags::COMPUTE,
                typ: vk::DescriptorType::STORAGE_IMAGE,
                count: 1,
            },
        )]
        .into_iter()
        .collect(),
    )?;

    let descriptor_set =
        DescriptorSet::new(descriptor_pool.clone(), descriptor_set_layout.clone(), 1)?[0].clone();
    descriptor_set
        .lock()
        .unwrap()
        .bind_storage_image(image.clone(), 0, 0, 1)?;

    println!("Shader compiling...");
    let mut reader = BufReader::new(std::fs::File::open("examples/shaders/path.comp")?);
    let mut source = String::new();
    reader.read_to_string(&mut source).unwrap();

    let compiler = shaderc::Compiler::new().unwrap();
    let binary = compiler.compile_into_spirv(
        &source,
        shaderc::ShaderKind::Compute,
        "particles.comp",
        "main",
        None,
    )?;

    let shader = Shader::new(
        device.clone(),
        CString::new("main")?,
        vk::ShaderStageFlags::COMPUTE,
        binary.as_binary().to_vec(),
    )?;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct PushConsts {
        time: f32,
        frame_num: u32,
        focal_dist: f32,
        samples_per_frame: i32,
    }

    unsafe impl bytemuck::Zeroable for PushConsts {}
    unsafe impl bytemuck::Pod for PushConsts {}

    let pipeline_layout = PipelineLayout::new(
        descriptor_pool,
        vec![descriptor_set_layout],
        &[vk::PushConstantRange::default()
            .size(size_of::<PushConsts>() as u32)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)],
    )?;
    let pipeline = Pipeline::new_compute(pipeline_layout, shader, None)?;

    println!("Done!\nRendering...");

    let mut command_buffer = CommandBufferBuilder::new(allocator.clone(), 0)?
        .bind_pipeline(pipeline)
        .bind_descriptor_sets(0, vec![descriptor_set])
        .push_constants(
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&PushConsts {
                time: SystemTime::UNIX_EPOCH.elapsed().unwrap().as_secs_f32(),
                frame_num: FRAMES,
                focal_dist: 1.,
                samples_per_frame: SAMPLES,
            }),
        )
        .dispatch([(width + 7) / 8, (height + 7) / 8, 1])
        .build(queue.clone())?;

    command_buffer.flush()?;
    command_buffer.wait()?;

    println!("Done!\nSaving...");

    let w = BufWriter::new(
        File::options()
            .write(true)
            .create(true)
            .open("examples/path/out.png")?,
    );

    let size = (width * height * 16) as u64;

    let buffer = crystal_vk::buffer::Buffer::<AnyBuffer>::new(
        device,
        BufferInfo {
            size,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            usage: vk::BufferUsageFlags::TRANSFER_DST,
            properties: vk::MemoryPropertyFlags::HOST_VISIBLE,
        },
    )?;

    let mut command_buffer = CommandBufferBuilder::new(allocator, 0)?
        .transition_image_layout(image.clone(), vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .copy_image_to_buffer(buffer.clone(), image)
        .build(queue)?;

    command_buffer.flush()?;
    command_buffer.wait()?;

    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Sixteen);
    let mut writer = encoder.write_header().unwrap();

    let mut lock = buffer.write().unwrap();
    let memory = lock.bind_memory(0..size)?;

    let mut data = Vec::new();

    for offset in (0..size).step_by(16) {
        let offset = offset as usize;
        let r: [u8; 4] = memory[offset..offset + 4].try_into().unwrap();
        let g: [u8; 4] = memory[offset + 4..offset + 8].try_into().unwrap();
        let b: [u8; 4] = memory[offset + 8..offset + 12].try_into().unwrap();

        let r = f32::from_le_bytes(r).clamp(0., 1.);
        let g = f32::from_le_bytes(g).clamp(0., 1.);
        let b = f32::from_le_bytes(b).clamp(0., 1.);

        let r = (r * 65535.0) as u16;
        let g = (g * 65535.0) as u16;
        let b = (b * 65535.0) as u16;

        data.push(r.to_le_bytes()[0]);
        data.push(r.to_le_bytes()[1]);
        data.push(g.to_le_bytes()[0]);
        data.push(g.to_le_bytes()[1]);
        data.push(b.to_le_bytes()[0]);
        data.push(b.to_le_bytes()[1]);
    }

    writer.write_image_data(&data).unwrap();

    println!("Done!");

    Ok(())
}
