use std::{
    error::Error,
    sync::{Arc, Mutex, RwLock},
};

use ash::vk;

use crate::{
    buffer::{AnyBuffer, Buffer, BufferInfo, IndexBuffer, VertexBuffer},
    device::Device,
};

pub struct GeometryTriangles {
    pub vertex_buffer: Arc<RwLock<Buffer<VertexBuffer>>>,
    pub vertex_stride: u64,
    pub vertex_format: vk::Format,
    pub vertex_max: u32,
    pub index_type: vk::IndexType,
    pub index_buffer: Arc<RwLock<Buffer<IndexBuffer>>>,
    pub index_count: u32,
}

impl GeometryTriangles {
    pub(crate) fn as_vk<'a>(&self) -> vk::AccelerationStructureGeometryKHR<'a> {
        let vertex_lock = self.vertex_buffer.read().unwrap();
        let index_lock = self.index_buffer.read().unwrap();

        let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(self.vertex_format)
            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: unsafe { vertex_lock.get_buffer_device_address() },
            })
            .vertex_stride(self.vertex_stride)
            .max_vertex(self.vertex_max)
            .index_type(self.index_type)
            .index_data(vk::DeviceOrHostAddressConstKHR {
                device_address: unsafe { index_lock.get_buffer_device_address() },
            });

        vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
            .flags(vk::GeometryFlagsKHR::OPAQUE)
    }
}

pub struct AccelerationStructure {
    handle: ash::khr::acceleration_structure::Device,
    geometries: Vec<GeometryTriangles>,
    pub(crate) blas: vk::AccelerationStructureKHR,
    pub(crate) blas_buffer: Option<Arc<RwLock<Buffer<AnyBuffer>>>>,
    pub(crate) scratch_buffer: Option<Arc<RwLock<Buffer<AnyBuffer>>>>,
    pub(crate) device: Arc<Device>,
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        // Destroy the Vulkan acceleration structure handle when this wrapper is dropped.
        let handle = ash::khr::acceleration_structure::Device::new(
            &self.device.instance.handle,
            &self.device.handle,
        );
        unsafe {
            handle.destroy_acceleration_structure(self.blas, None);
        }
        // `self._buffer` will be dropped automatically, freeing the underlying GPU memory.
    }
}

impl AccelerationStructure {
    pub fn new_triangles(
        device: Arc<Device>,
        geometries: Vec<GeometryTriangles>,
    ) -> Result<Arc<Mutex<Self>>, Box<dyn Error>> {
        let handle =
            ash::khr::acceleration_structure::Device::new(&device.instance.handle, &device.handle);

        Ok(Arc::new(Mutex::new(Self {
            handle: handle,
            geometries,
            blas: vk::AccelerationStructureKHR::null(),
            blas_buffer: None,
            scratch_buffer: None,
            device,
        })))
    }

    pub(crate) fn build(
        &mut self,
        command_buffer: vk::CommandBuffer,
    ) -> Result<(), Box<dyn Error>> {
        let geometries = &self.geometries;

        let g = geometries.iter().map(|g| g.as_vk()).collect::<Vec<_>>();

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(&g);

        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();

        // TODO Triangles only!
        let primitive_counts = geometries
            .iter()
            .map(|g| g.index_count / 3)
            .collect::<Vec<_>>();
        unsafe {
            self.handle.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &primitive_counts,
                &mut sizes,
            )
        };

        // Allocate a buffer that backs the acceleration structure. This buffer must be retained for the
        // lifetime of the `AccelerationStructure` instance.
        let blas_buffer = Buffer::<AnyBuffer>::new(
            self.device.clone(),
            BufferInfo {
                size: sizes.acceleration_structure_size,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            },
        )?;

        self.blas_buffer = Some(blas_buffer.clone());

        let lock = blas_buffer.write().unwrap();

        let blas_create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(lock.handle)
            .size(sizes.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
        let blas = unsafe {
            self.handle
                .create_acceleration_structure(&blas_create_info, None)?
        };

        self.blas = blas;

        // Release the temporary lock; the buffer itself is kept alive via the `blas_buffer` Arc.
        drop(lock);

        let scratch_buffer = Buffer::<AnyBuffer>::new(
            self.device.clone(),
            BufferInfo {
                size: sizes.build_scratch_size,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                properties: vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::HOST_VISIBLE,
            },
        )?;

        self.scratch_buffer = Some(scratch_buffer.clone());

        let lock = scratch_buffer.read().unwrap();

        let build_info =
            build_info
                .dst_acceleration_structure(blas)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: unsafe { lock.get_buffer_device_address() },
                });
        let range_infos: Vec<_> = primitive_counts
            .iter()
            .map(|count| {
                vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(*count)
            })
            .collect();

        unsafe {
            self.handle.cmd_build_acceleration_structures(
                command_buffer,
                &[build_info],
                &[&range_infos],
            );
        }

        Ok(())
    }
}
