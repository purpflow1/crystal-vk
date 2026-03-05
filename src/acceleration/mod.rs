use std::{
    error::Error,
    sync::{Arc, Mutex},
};

use ash::vk;

use crate::{
    buffer::{AnyBuffer, Buffer, BufferInfo},
    command::{CommandBufferAllocator, command_buffer_builder::CommandBufferBuilder},
    device::{Device, queue::Queue},
};

pub struct AccelerationStructure {
    pub(crate) blas: vk::AccelerationStructureKHR,
    pub(crate) device: Arc<Device>,
}

impl AccelerationStructure {
    pub fn build(
        command_allocator: Arc<CommandBufferAllocator>,
        queue_family_index: u32,
        queue: Arc<Mutex<Queue>>,
        geometries: Vec<(vk::AccelerationStructureGeometryKHR, u32)>,
    ) -> Result<Arc<Self>, Box<dyn Error>> {
        let device = command_allocator.device.clone();

        let handle =
            ash::khr::acceleration_structure::Device::new(&device.instance.handle, &device.handle);

        let g = geometries.iter().map(|g| g.0).collect::<Vec<_>>();

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(&g);

        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();

        let primitive_counts = geometries.iter().map(|g| g.1).collect::<Vec<_>>();
        unsafe {
            handle.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &primitive_counts,
                &mut sizes,
            )
        };

        let blas_buffer = Buffer::<AnyBuffer>::new(
            device.clone(),
            BufferInfo {
                size: sizes.acceleration_structure_size,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            },
        )?;
        let lock = blas_buffer.write().unwrap();

        let blas_create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(lock.handle)
            .size(sizes.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
        let blas = unsafe { handle.create_acceleration_structure(&blas_create_info, None)? };

        drop(lock);

        let scratch_buffer = Buffer::<AnyBuffer>::new(
            device.clone(),
            BufferInfo {
                size: sizes.build_scratch_size,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            },
        )?;
        let mut lock = scratch_buffer.write().unwrap();
        lock.bind_memory(0..sizes.build_scratch_size)?;

        let build_info =
            build_info
                .dst_acceleration_structure(blas)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: lock.mapped as u64,
                });
        let range_infos: Vec<_> = primitive_counts
            .iter()
            .map(|count| {
                vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(*count)
            })
            .collect();

        let builder = CommandBufferBuilder::new(command_allocator, queue_family_index)?;
        unsafe {
            handle.cmd_build_acceleration_structures(
                builder.handle,
                &[build_info],
                &[&range_infos],
            );
        }

        let mut command_buffer = builder.build(queue)?;
        command_buffer.flush()?;
        command_buffer.wait()?;

        Ok(Arc::new(Self { blas, device }))
    }
}
