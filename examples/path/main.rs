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
use half::f16;

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
        vk::Format::R16G16B16A16_SFLOAT,
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
        .dispatch([(width.div_ceil(8) + 7), (height.div_ceil(8) + 7), 1])
        .build(queue.clone())?;

    command_buffer.flush()?;
    command_buffer.wait()?;

    println!("Done!\nSaving...");

    let w = BufWriter::new(
        File::options()
            .write(true)
            .create(true)
            .truncate(true)
            .open("examples/path/out.png")?,
    );

    let size = (width * height * 8) as u64;

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
    let mem = lock.bind_memory(0..size)?;

    let mut data = Vec::new();

    for offset in (0..size).step_by(8) {
        let offset = offset as usize;
        // Each component is stored as a 16‑bit half‑float (little‑endian)
        let r16 = u16::from_le_bytes(mem[offset..offset + 2].try_into().unwrap());
        let g16 = u16::from_le_bytes(mem[offset + 2..offset + 4].try_into().unwrap());
        let b16 = u16::from_le_bytes(mem[offset + 4..offset + 6].try_into().unwrap());
        // Alpha is ignored (mem[offset + 6..offset + 8])

        let rf = f16::from_bits(r16).to_f32().clamp(0.0, 1.0);
        let gf = f16::from_bits(g16).to_f32().clamp(0.0, 1.0);
        let bf = f16::from_bits(b16).to_f32().clamp(0.0, 1.0);

        data.extend_from_slice(&((rf * 65535.0) as u16).to_le_bytes());
        data.extend_from_slice(&((gf * 65535.0) as u16).to_le_bytes());
        data.extend_from_slice(&((bf * 65535.0) as u16).to_le_bytes());
    }

    writer.write_image_data(&data).unwrap();

    println!("Done!");

    Ok(())
}
