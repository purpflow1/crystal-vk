use std::{error::Error, ops::Range, sync::Arc};

use ash::vk;

use crate::device::Device;

#[derive(Clone, Copy)]
pub(super) struct BufferInfo {
    pub size: u64,
    pub sharing_mode: vk::SharingMode,
    pub usage: vk::BufferUsageFlags,
    pub properties: vk::MemoryPropertyFlags,
}

pub(super) struct BufferData {
    pub(super) handle: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    pub(super) info: BufferInfo,

    device: Arc<Device>,
}

impl Drop for BufferData {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.free_memory(self.memory, None);
            self.device.handle.destroy_buffer(self.handle, None);
        }
    }
}

impl BufferData {
    pub(super) fn new(device: Arc<Device>, info: BufferInfo) -> Result<Self, Box<dyn Error>> {
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

        let mapped = unsafe {
            device
                .handle
                .map_memory(device_memory, 0, info.size, vk::MemoryMapFlags::empty())
        }? as *mut u8;

        Ok(Self {
            handle: buffer,
            memory: device_memory,
            mapped,
            info,
            device,
        })
    }

    pub(crate) unsafe fn get_memory<'a>(&self, range: Range<u64>) -> &'a mut [u8] {
        unsafe {
            let ptr = self.mapped.byte_add(range.start as usize);
            std::slice::from_raw_parts_mut(ptr, (range.end - range.start) as usize)
        }
    }
}
