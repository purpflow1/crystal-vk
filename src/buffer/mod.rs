use std::{
    error::Error,
    marker::PhantomData,
    ops::Range,
    sync::{Arc, RwLock},
};

use ash::vk;

use crate::device::Device;

pub trait BufferUsage {}

pub struct VertexBuffer {}
impl BufferUsage for VertexBuffer {}

pub struct IndexBuffer {}
impl BufferUsage for IndexBuffer {}

pub struct InderectBuffer {}
impl BufferUsage for InderectBuffer {}

pub struct AnyBuffer {}
impl BufferUsage for AnyBuffer {}

#[derive(Clone, Copy)]
pub struct BufferInfo {
    pub size: u64,
    pub sharing_mode: vk::SharingMode,
    pub usage: vk::BufferUsageFlags,
    pub properties: vk::MemoryPropertyFlags,
}

pub struct Buffer<Usage: BufferUsage + 'static> {
    pub(super) handle: vk::Buffer,
    memory: vk::DeviceMemory,
    pub info: BufferInfo,
    pub mapped: *mut u8,

    pub device: Arc<Device>,

    _usage: PhantomData<Usage>,
}

unsafe impl<Usage: BufferUsage> Send for Buffer<Usage> {}
unsafe impl<Usage: BufferUsage> Sync for Buffer<Usage> {}

impl<Usage: BufferUsage> Drop for Buffer<Usage> {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.free_memory(self.memory, None);
            self.device.handle.destroy_buffer(self.handle, None);
        }
    }
}

impl<Usage: BufferUsage> Buffer<Usage> {
    pub fn new(device: Arc<Device>, info: BufferInfo) -> Result<Arc<RwLock<Self>>, Box<dyn Error>> {
        let create_info = vk::BufferCreateInfo::default()
            .size(info.size)
            .usage(info.usage)
            .sharing_mode(info.sharing_mode);

        let buffer = unsafe { device.handle.create_buffer(&create_info, None) }?;

        let memory_requirements = unsafe { device.handle.get_buffer_memory_requirements(buffer) };

        let memory_type_index = device
            .physical_device
            .find_memory_type_index(info.properties, memory_requirements.memory_type_bits)?;

        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index);

        let device_memory = unsafe { device.handle.allocate_memory(&memory_allocate_info, None) }?;

        unsafe { device.handle.bind_buffer_memory(buffer, device_memory, 0) }?;

        Ok(Arc::new(RwLock::new(Self {
            handle: buffer,
            memory: device_memory,
            mapped: std::ptr::null_mut(),
            info,
            device,
            _usage: PhantomData,
        })))
    }

    pub fn bind_memory<'a>(&mut self, range: Range<u64>) -> Result<&'a mut [u8], Box<dyn Error>> {
        if !self.mapped.is_null() {
            unsafe { self.device.handle.unmap_memory(self.memory) };
            self.mapped = std::ptr::null_mut();
        }

        let mapped = unsafe {
            self.device.handle.map_memory(
                self.memory,
                range.start,
                range.end - range.start,
                vk::MemoryMapFlags::empty(),
            )
        }? as *mut u8;

        self.mapped = mapped;

        let slice =
            unsafe { std::slice::from_raw_parts_mut(mapped, (range.end - range.start) as usize) };

        Ok(slice)
    }

    pub fn get_strided_device_addr_region(
        &self,
        offset: u64,
        size: u64,
        stride: u64,
    ) -> vk::StridedDeviceAddressRegionKHR {
        let addr = unsafe { self.get_buffer_device_address() };

        vk::StridedDeviceAddressRegionKHR::default()
            .device_address(addr + offset)
            .size(size)
            .stride(stride)
    }

    pub unsafe fn get_buffer_device_address(&self) -> u64 {
        let address_info = vk::BufferDeviceAddressInfo::default().buffer(self.handle);
        unsafe { self.device.handle.get_buffer_device_address(&address_info) }
    }
}
