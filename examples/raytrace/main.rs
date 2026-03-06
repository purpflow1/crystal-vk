use std::{
    error::Error,
    ffi::CString,
    fs::File,
    io::{BufReader, BufWriter, Read},
    mem::size_of,
    sync::Arc,
    time::SystemTime,
};

use ash::vk;
use crystal_vk::{
    acceleration::{AccelerationStructure, GeometryTriangles},
    buffer::{AnyBuffer, Buffer, BufferInfo, IndexBuffer, VertexBuffer},
    command::{CommandBufferAllocator, command_buffer_builder::CommandBufferBuilder},
    device::Device,
    image,
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
    let width = 800;
    let height = 600;

    // -------------------------------------------------
    // Vulkan instance / device
    // -------------------------------------------------
    let instance = crystal_vk::instance::Instance::new()?;
    let physical_device = instance.enumerate_physical_devices(None)?[0].clone();

    let rt_props = physical_device.info.rt_props;

    let (device, queues) = crystal_vk::device::Device::new(
        physical_device,
        vk::PhysicalDeviceFeatures::default(),
        vec![
            vk::KHR_DEFERRED_HOST_OPERATIONS_NAME,
            vk::KHR_ACCELERATION_STRUCTURE_NAME,
            vk::KHR_RAY_TRACING_PIPELINE_NAME,
        ],
    )?;

    // -------------------------------------------------
    // Extension check
    // -------------------------------------------------
    if !device.extensions.contains(
        &vk::KHR_ACCELERATION_STRUCTURE_NAME
            .to_str()
            .unwrap()
            .to_string(),
    ) {
        return Err("Device does not support acceleration structure!".into());
    }
    if !device.extensions.contains(
        &vk::KHR_RAY_TRACING_PIPELINE_NAME
            .to_str()
            .unwrap()
            .to_string(),
    ) {
        return Err("Device does not support ray tracing pipeline!".into());
    }

    // -------------------------------------------------
    // Geometry (vertex)
    // -------------------------------------------------
    let vertices: Vec<f32> = vec![
        // Pos           // Color
        0.0, -0.5, 0.0, 1.0, 0.0, 0.0, // V0 – красный
        0.5, 0.5, 0.0, 0.0, 1.0, 0.0, // V1 – зелёный
        -0.5, 0.5, 0.0, 0.0, 0.0, 1.0, // V2 – синий
    ];
    let indices: Vec<u32> = vec![0, 1, 2];

    // --- vertex buffer ---------------------------------------------------------
    let vertex_buffer = Buffer::<VertexBuffer>::new(
        device.clone(),
        BufferInfo {
            size: (vertices.len() * size_of::<f32>()) as u64,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            properties: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
        },
    )?;
    {
        let mut lock = vertex_buffer.write().unwrap();
        let size = lock.info.size;
        let mem = lock.bind_memory(0..size)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                vertices.as_ptr() as *const u8,
                mem.as_mut_ptr(),
                size as usize,
            );
        }
    }

    // --- index buffer ---------------------------------------------------------
    let index_buffer = Buffer::<IndexBuffer>::new(
        device.clone(),
        BufferInfo {
            size: (indices.len() * size_of::<u32>()) as u64,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            usage: vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            properties: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
        },
    )?;
    {
        let mut lock = index_buffer.write().unwrap();
        let size = lock.info.size;
        let mem = lock.bind_memory(0..size)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                indices.as_ptr() as *const u8,
                mem.as_mut_ptr(),
                size as usize,
            );
        }
    }

    // -------------------------------------------------------------------------
    // BLAS – acceleration structure
    // -------------------------------------------------------------------------
    let acceleration_structure = AccelerationStructure::new_triangles(
        device.clone(),
        vec![GeometryTriangles {
            vertex_buffer: vertex_buffer.clone(),
            vertex_stride: (6 * size_of::<f32>()) as u64, // 3 pos + 3 color
            vertex_format: vk::Format::R32G32B32_SFLOAT,
            vertex_max: 3,
            index_type: vk::IndexType::UINT32,
            index_buffer: index_buffer.clone(),
            index_count: 3,
        }],
    )?;

    // -------------------------------------------------------------------------
    // Queue and command allocator
    // -------------------------------------------------------------------------
    let queue = queues.first_key_value().unwrap().1[0].clone();
    let allocator = CommandBufferAllocator::new(queues.clone())?;

    // -------------------------------------------------------------------------
    // Storage image
    // -------------------------------------------------------------------------
    let image = image::Image::new(
        device.clone(),
        [width, height],
        vk::Format::R32G32B32A32_SFLOAT,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
    )?;

    // -------------------------------------------------------------------------
    // Descriptor pool and set
    // -------------------------------------------------------------------------
    let descriptor_pool = DescriptorPool::new(
        device.clone(),
        &[vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)],
    )?;

    // layout for storage image + TLAS
    let descriptor_set_layout = DescriptorSetLayout::new(
        device.clone(),
        vec![
            // 0 – storage image
            (
                0,
                LayoutAllocInfo {
                    stages: vk::ShaderStageFlags::RAYGEN_KHR,
                    typ: vk::DescriptorType::STORAGE_IMAGE,
                    count: 1,
                },
            ),
            (
                1,
                LayoutAllocInfo {
                    stages: vk::ShaderStageFlags::RAYGEN_KHR,
                    typ: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                    count: 1,
                },
            ),
        ]
        .into_iter()
        .collect(),
    )?;

    let descriptor_set =
        DescriptorSet::new(descriptor_pool.clone(), descriptor_set_layout.clone(), 1)?[0].clone();

    {
        let mut lock = descriptor_set.lock().unwrap();
        lock.bind_storage_image(image.clone(), 0, 0, 1)?;
        lock.bind_acceleration_structure(acceleration_structure, 1)?;
    }

    // -------------------------------------------------------------------------
    // Compiling ray‑tracing shaders
    // -------------------------------------------------------------------------
    fn compile_shader(
        device: Arc<Device>,
        path: &str,
        kind: shaderc::ShaderKind,
        entry: &str,
    ) -> Result<Arc<Shader>, Box<dyn Error>> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut source = String::new();
        reader.read_to_string(&mut source)?;

        let compiler = shaderc::Compiler::new().unwrap();
        let binary = compiler.compile_into_spirv(&source, kind, path, entry, None)?;

        Ok(Shader::new(
            device,
            CString::new(entry)?,
            match kind {
                shaderc::ShaderKind::RayGeneration => vk::ShaderStageFlags::RAYGEN_KHR,
                shaderc::ShaderKind::Miss => vk::ShaderStageFlags::MISS_KHR,
                shaderc::ShaderKind::ClosestHit => vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                _ => vk::ShaderStageFlags::RAYGEN_KHR, // fallback – not used
            },
            binary.as_binary().to_vec(),
        )?)
    }

    let raygen_shader = compile_shader(
        device.clone(),
        "examples/shaders/raygen.rgen",
        shaderc::ShaderKind::RayGeneration,
        "main",
    )?;
    let miss_shader = compile_shader(
        device.clone(),
        "examples/shaders/miss.rmiss",
        shaderc::ShaderKind::Miss,
        "main",
    )?;
    let hit_shader = compile_shader(
        device.clone(),
        "examples/shaders/closesthit.rchit",
        shaderc::ShaderKind::ClosestHit,
        "main",
    )?;

    // -------------------------------------------------------------------------
    // Pipeline layout (push‑constants + descriptor set)
    // -------------------------------------------------------------------------
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
        descriptor_pool.clone(),
        vec![descriptor_set_layout.clone()],
        &[vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
            .offset(0)
            .size(size_of::<PushConsts>() as u32)],
    )?;

    // -------------------------------------------------------------------------
    // Ray‑tracing pipeline
    // -------------------------------------------------------------------------
    let rt_pipeline = Pipeline::new_raytrace(
        pipeline_layout.clone(),
        vec![raygen_shader, miss_shader, hit_shader],
        None,
        None,
    )?;

    // -------------------------------------------------------------------------
    // Shader Binding Table (SBT)
    // -------------------------------------------------------------------------

    let handle_size = rt_props.shader_group_handle_size as usize;
    let handle_alignment = rt_props.shader_group_handle_alignment as usize;
    let group_count = 3; // 3 шейдера

    // Alignment
    let aligned_handle_size =
        ((handle_size + handle_alignment - 1) / handle_alignment) * handle_alignment;

    let sbt_size = (aligned_handle_size * group_count) as vk::DeviceSize;

    // SBT Buffer
    let sbt_buffer = Buffer::<AnyBuffer>::new(
        device.clone(),
        BufferInfo {
            size: sbt_size,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            usage: vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::TRANSFER_SRC,
            properties: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
        },
    )?;

    // Descriptors and groups
    {
        // Get all the handles
        let shader_handles = rt_pipeline.get_raytrace_shader_groups(
            0,
            group_count as u32,
            handle_size * group_count,
        )?;

        // Copy to buffer
        let mut lock = sbt_buffer.write().unwrap();
        let ptr = lock.bind_memory(0..sbt_size)?;
        for (i, chunk) in shader_handles.chunks(handle_size).enumerate() {
            let dst_offset = i * aligned_handle_size;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    chunk.as_ptr(),
                    ptr.as_mut_ptr().add(dst_offset),
                    handle_size,
                );
            }
        }
    }

    let lock = sbt_buffer.read().unwrap();

    // Regions
    let raygen_region =
        lock.get_strided_device_addr_region(0, handle_size as u64, aligned_handle_size as u64);
    let miss_region = lock.get_strided_device_addr_region(
        aligned_handle_size as u64,
        handle_size as u64,
        aligned_handle_size as u64,
    );
    let hit_region = lock.get_strided_device_addr_region(
        2 * aligned_handle_size as u64,
        handle_size as u64,
        aligned_handle_size as u64,
    );

    drop(lock);

    let callable_region = vk::StridedDeviceAddressRegionKHR::default();

    // -------------------------------------------------------------------------
    // Rendering
    // -------------------------------------------------------------------------
    println!("Rendering…");

    let builder = CommandBufferBuilder::new(allocator.clone(), 0)?
        .transition_image_layout(image.clone(), vk::ImageLayout::GENERAL)
        .bind_pipeline(rt_pipeline.clone())
        .bind_descriptor_sets(0, vec![descriptor_set.clone()])
        .push_constants(
            vk::ShaderStageFlags::RAYGEN_KHR,
            0,
            bytemuck::bytes_of(&PushConsts {
                time: SystemTime::UNIX_EPOCH.elapsed().unwrap().as_secs_f32(),
                frame_num: FRAMES,
                focal_dist: 1.0,
                samples_per_frame: SAMPLES,
            }),
        )
        .trace_rays(
            &raygen_region,
            &miss_region,
            &hit_region,
            &callable_region,
            [width.div_ceil(8) as u32, height.div_ceil(8) as u32, 1],
        );

    let mut cmd = builder.build(queue.clone())?;

    cmd.flush()?;
    cmd.wait()?;

    // -------------------------------------------------------------------------
    // Saving
    // -------------------------------------------------------------------------
    println!("Saving image…");
    let w = BufWriter::new(
        File::options()
            .write(true)
            .create(true)
            .truncate(true)
            .open("examples/raytrace/out.png")?,
    );

    let size = (width * height * 16) as u64; // R32G32B32A32_SFLOAT
    let buffer = crystal_vk::buffer::Buffer::<AnyBuffer>::new(
        device.clone(),
        BufferInfo {
            size,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            usage: vk::BufferUsageFlags::TRANSFER_DST,
            properties: vk::MemoryPropertyFlags::HOST_VISIBLE,
        },
    )?;

    // копируем изображение в буфер
    let mut copy_cmd = CommandBufferBuilder::new(allocator, 0)?
        .transition_image_layout(image.clone(), vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .copy_image_to_buffer(buffer.clone(), image.clone())
        .build(queue.clone())?;

    copy_cmd.flush()?;
    copy_cmd.wait()?;

    // пишем PNG
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Sixteen);
    let mut writer = encoder.write_header().unwrap();

    let mut lock = buffer.write().unwrap();
    let mem = lock.bind_memory(0..size)?;
    let mut data = Vec::with_capacity((width * height * 3 * 2) as usize);

    for offset in (0..size).step_by(16) {
        let offset = offset as usize;
        let r: [u8; 4] = mem[offset..offset + 4].try_into().unwrap();
        let g: [u8; 4] = mem[offset + 4..offset + 8].try_into().unwrap();
        let b: [u8; 4] = mem[offset + 8..offset + 12].try_into().unwrap();

        let rf = f32::from_le_bytes(r).clamp(0.0, 1.0);
        let gf = f32::from_le_bytes(g).clamp(0.0, 1.0);
        let bf = f32::from_le_bytes(b).clamp(0.0, 1.0);

        data.extend_from_slice(&((rf * 65535.0) as u16).to_le_bytes());
        data.extend_from_slice(&((gf * 65535.0) as u16).to_le_bytes());
        data.extend_from_slice(&((bf * 65535.0) as u16).to_le_bytes());
    }

    writer.write_image_data(&data).unwrap();
    println!("Finished!");
    Ok(())
}
