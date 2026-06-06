use std::{
    collections::VecDeque,
    error::Error,
    ops::{Bound, RangeBounds},
    sync::{Arc, RwLock},
};

use ash::vk;

use crate::device::Device;

struct MappedMemoryRange {
    size: u64,
    offset: u64,
}

#[derive(Clone, Copy)]
pub struct BufferInfo {
    pub size: u64,
    pub sharing_mode: vk::SharingMode,
    pub usage: vk::BufferUsageFlags,
    pub properties: vk::MemoryPropertyFlags,
}

pub struct Buffer {
    pub(super) handle: vk::Buffer,
    memory: vk::DeviceMemory,
    pub info: BufferInfo,
    pub mapped: *mut u8,
    cached_ranges: VecDeque<MappedMemoryRange>,

    pub device: Arc<Device>,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.free_memory(self.memory, None);
            self.device.handle.destroy_buffer(self.handle, None);
        }
    }
}

impl Buffer {
    pub fn new(device: Arc<Device>, info: BufferInfo) -> Result<Arc<RwLock<Self>>, Box<dyn Error>> {
        let create_info = vk::BufferCreateInfo::default()
            .size(info.size)
            .usage(info.usage)
            .sharing_mode(info.sharing_mode);

        let buffer = unsafe { device.handle.create_buffer(&create_info, None) }?;

        let memory_requirements = unsafe { device.handle.get_buffer_memory_requirements(buffer) };

        let memory_type_index = device
            .physical_device
            .find_memory_type_index(info.properties, memory_requirements.memory_type_bits);

        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index);

        let device_memory = unsafe { device.handle.allocate_memory(&memory_allocate_info, None) }?;

        unsafe { device.handle.bind_buffer_memory(buffer, device_memory, 0) }?;

        Ok(Arc::new(RwLock::new(Self {
            handle: buffer,
            memory: device_memory,
            mapped: std::ptr::null_mut(),
            cached_ranges: VecDeque::new(),
            info,
            device,
        })))
    }

    pub fn flush(&self) -> Result<(), Box<dyn Error>> {
        let ranges: Vec<_> = self
            .cached_ranges
            .iter()
            .map(|range| {
                vk::MappedMemoryRange::default()
                    .memory(self.memory)
                    .offset(range.offset)
                    .size(range.size)
            })
            .collect();

        unsafe { self.device.handle.flush_mapped_memory_ranges(&ranges)? };

        Ok(())
    }

    pub fn invalidate(&self) -> Result<(), Box<dyn Error>> {
        let ranges: Vec<_> = self
            .cached_ranges
            .iter()
            .map(|range| {
                vk::MappedMemoryRange::default()
                    .memory(self.memory)
                    .offset(range.offset)
                    .size(range.size)
            })
            .collect();

        unsafe {
            self.device
                .handle
                .invalidate_mapped_memory_ranges(&ranges)?
        };

        Ok(())
    }

    pub fn bind_memory<'a>(
        &mut self,
        range: impl RangeBounds<u64>,
    ) -> Result<&'a mut [u8], Box<dyn Error>> {
        if !self.mapped.is_null() {
            unsafe { self.device.handle.unmap_memory(self.memory) };
            self.mapped = std::ptr::null_mut();
        }

        let start = match range.start_bound() {
            Bound::Included(bound) => *bound,
            Bound::Excluded(bound) => *bound + 1,
            _ => 0,
        };

        let end = match range.end_bound() {
            Bound::Included(bound) => *bound + 1,
            Bound::Excluded(bound) => *bound,
            _ => self.info.size,
        };

        let bounds_len = end - start;

        let mapped = unsafe {
            self.device.handle.map_memory(
                self.memory,
                start,
                bounds_len,
                vk::MemoryMapFlags::empty(),
            )
        }? as *mut u8;

        self.mapped = mapped;

        if !self
            .info
            .properties
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            self.cached_ranges.push_back(MappedMemoryRange {
                size: bounds_len,
                offset: start,
            });
        }

        let slice = unsafe { std::slice::from_raw_parts_mut(mapped, bounds_len as usize) };

        Ok(slice)
    }
}
