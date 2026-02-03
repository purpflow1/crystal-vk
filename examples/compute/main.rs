use std::{
    collections::BTreeMap,
    error::Error,
    ffi::CString,
    fs::File,
    io::{BufReader, Read},
};

use crystal_vk::{
    buffer::{Buffer, BufferInfo},
    command::{CommandBufferAllocator, command_buffer_builder::CommandBufferBuilder},
    device::Device,
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
    vk,
};
use futures::executor;

fn main() -> Result<(), Box<dyn Error>> {
    let (device, queues) = Device::compute(|devices| devices[0].clone())?;

    let buffer_in = Buffer::new(
        device.clone(),
        BufferInfo {
            size: 512,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            properties: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
        },
    )?;

    let buffer_out = Buffer::new(
        device.clone(),
        BufferInfo {
            size: 512,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            properties: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
        },
    )?;

    let mut layout_infos = BTreeMap::new();
    layout_infos.insert(
        0,
        LayoutAllocInfo {
            stages: vk::ShaderStageFlags::COMPUTE,
            typ: vk::DescriptorType::STORAGE_BUFFER,
            count: 1,
        },
    );
    layout_infos.insert(
        1,
        LayoutAllocInfo {
            stages: vk::ShaderStageFlags::COMPUTE,
            typ: vk::DescriptorType::STORAGE_BUFFER,
            count: 1,
        },
    );

    let descriptor_pool = DescriptorPool::new(device.clone(), 2)?;
    let descriptor_set_layout = DescriptorSetLayout::new(device.clone(), layout_infos)?;
    let descriptor_set =
        DescriptorSet::new(descriptor_pool.clone(), descriptor_set_layout.clone(), 1)?[0].clone();

    let mut reader = BufReader::new(File::open("examples/shaders/plain.comp")?);
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

    let pipeline_layout = PipelineLayout::new(descriptor_pool, vec![descriptor_set_layout])?;
    let pipeline = Pipeline::new_compute(pipeline_layout.clone(), shader)?;

    let mut lock = buffer_in.write().unwrap();
    let memory = lock.bind_memory(0..512).unwrap();
    memory.fill(0);
    drop(lock);

    let (queue_info, queue) = queues
        .iter()
        .find_map(|(info, queues)| {
            if info.flags.intersects(vk::QueueFlags::COMPUTE) {
                Some((*info, queues[0].clone()))
            } else {
                None
            }
        })
        .unwrap();

    let command_buffer_allocator = CommandBufferAllocator::new(queues)?;
    let command_buffer_builder = CommandBufferBuilder::new(
        command_buffer_allocator,
        queue_info.index,
        vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
    )?;

    {
        let mut lock = descriptor_set.lock().unwrap();
        lock.bind_buffer(buffer_in, 0, 0, 1)?;
        lock.bind_buffer(buffer_out.clone(), 1, 0, 1)?;
    }

    let command_buffer_builder = command_buffer_builder
        .bind_pipeline(pipeline)
        .bind_descriptor_sets(0, vec![descriptor_set.clone()])
        .unwrap()
        .dispatch([1, 1, 1])
        .unwrap();

    let future = command_buffer_builder.build(queue)?;
    executor::block_on(future)?;

    Ok(())
}
