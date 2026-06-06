use std::{
    collections::BTreeMap,
    error::Error,
    ffi::CString,
    fs::File,
    io::{BufReader, Read},
    sync::Mutex,
    time::SystemTime,
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

static RNG_STATE: Mutex<u64> = Mutex::new(0);
const BUFFER_SIZE: u64 = 8192;

pub fn rand_u64() -> u64 {
    let mut s = RNG_STATE.lock().unwrap();
    let mut x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1Du64)
}

pub fn rand_f32() -> f32 {
    let v = (rand_u64() >> 40) as u32;
    (v as f32) / ((1u32 << 24) as f32)
}

fn main() -> Result<(), Box<dyn Error>> {
    *RNG_STATE.lock().unwrap() = SystemTime::UNIX_EPOCH.elapsed().unwrap().as_secs();

    let instance = crystal_vk::instance::Instance::new()?;
    let physical_device = instance
        .get_default_physical_device(None)
        .expect("error during enumerate device")
        .expect("cannot find default device");
    let (device, queues) = Device::new(
        physical_device,
        vk::PhysicalDeviceFeatures::default(),
        vec![
            vk::KHR_DEFERRED_HOST_OPERATIONS_NAME,
            #[cfg(target_os = "macos")]
            vk::KHR_PORTABILITY_SUBSET_NAME,
        ],
    )?;

    let buffer_in = Buffer::new(
        device.clone(),
        BufferInfo {
            size: BUFFER_SIZE,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            properties: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
        },
    )?;

    let buffer_out = Buffer::new(
        device.clone(),
        BufferInfo {
            size: BUFFER_SIZE,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
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

    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(2)];
    let descriptor_pool = DescriptorPool::new(device.clone(), &pool_sizes)?;
    let descriptor_set_layout = DescriptorSetLayout::new(device.clone(), layout_infos)?;
    let descriptor_set =
        DescriptorSet::new(descriptor_pool.clone(), descriptor_set_layout.clone(), 1)?[0].clone();

    let mut reader = BufReader::new(File::open("examples/shaders/biquad.comp")?);
    let mut source = String::new();
    reader.read_to_string(&mut source).unwrap();

    let compiler = shaderc::Compiler::new().unwrap();
    let binary = compiler.compile_into_spirv(
        &source,
        shaderc::ShaderKind::Compute,
        "biquad.comp",
        "main",
        None,
    )?;

    let shader = Shader::new(
        device.clone(),
        CString::new("main")?,
        vk::ShaderStageFlags::COMPUTE,
        binary.as_binary().to_vec(),
    )?;

    let pipeline_layout = PipelineLayout::new(descriptor_pool, vec![descriptor_set_layout], &[])?;
    let pipeline = Pipeline::new_compute(pipeline_layout.clone(), shader.clone(), None)?;

    {
        let mut lock = buffer_in.write().unwrap();
        let memory = lock.bind_memory(0..BUFFER_SIZE).unwrap();
        let memory: &mut [f32] = bytemuck::cast_slice_mut(memory);
        for word in memory {
            *word = rand_f32();
        }
    }

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
    let command_buffer_builder =
        CommandBufferBuilder::new(command_buffer_allocator, queue_info.index)?;

    {
        let mut lock = descriptor_set.lock().unwrap();
        lock.bind_buffer(buffer_in, 0, 0, 1)?;
        lock.bind_buffer(buffer_out.clone(), 1, 0, 1)?;
    }

    let command_buffer_builder = command_buffer_builder
        .bind_pipeline(pipeline)
        .bind_descriptor_sets(0, vec![descriptor_set.clone()])
        .dispatch([BUFFER_SIZE as u32 / 256, 1, 1]);

    let mut future = command_buffer_builder.build(queue)?;
    future.flush().unwrap();
    future.wait().unwrap();

    let mut lock = buffer_out.write().unwrap();
    let memory = lock.bind_memory(0..size_of::<f32>() as u64 * 8)?;
    let data: &[f32] = bytemuck::cast_slice(memory);
    dbg!(data);

    Ok(())
}
