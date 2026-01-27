mod memory;

use std::{
    error::Error,
    marker::PhantomData,
    ops::{Index, IndexMut, Range, RangeFrom, RangeFull, RangeTo},
    sync::{Arc, RwLock},
};

use ash::vk;

use crate::{
    buffer::memory::{BufferData, BufferInfo},
    device::Device,
};

pub struct BufferCreateInfo {
    pub len: u64,
    pub sharing_mode: vk::SharingMode,
    pub usage: vk::BufferUsageFlags,
    pub properties: vk::MemoryPropertyFlags,
}

/// Stores GPU data.
/// Can be indexed with `u64` only!
pub struct Buffer<T> {
    inner: BufferData,
    pub device: Arc<Device>,
    _t: PhantomData<T>,
}

impl<T> Buffer<T> {
    pub fn new(
        device: Arc<Device>,
        info: BufferCreateInfo,
    ) -> Result<Arc<RwLock<Self>>, Box<dyn Error>> {
        Ok(Arc::new(RwLock::new(Self {
            inner: BufferData::new(
                device.clone(),
                BufferInfo {
                    size: info.len * size_of::<T>() as u64,
                    sharing_mode: info.sharing_mode,
                    usage: info.usage,
                    properties: info.properties,
                },
            )?,
            device,
            _t: PhantomData,
        })))
    }

    pub(crate) fn as_raw(&self) -> vk::Buffer {
        self.inner.handle
    }

    /// Returns true if buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns length of buffer
    #[inline]
    pub fn len(&self) -> u64 {
        self.inner.info.size / size_of::<T>() as u64
    }
}

impl<T> Index<u64> for Buffer<T> {
    type Output = T;
    #[inline]
    fn index(&self, index: u64) -> &Self::Output {
        let size = size_of::<T>() as u64;
        let offset = index * size;
        let bytes = unsafe { self.inner.get_memory(offset..offset + size) };
        unsafe { &*((bytes.as_ptr()) as *const T) }
    }
}

impl<T> IndexMut<u64> for Buffer<T> {
    #[inline]
    fn index_mut(&mut self, index: u64) -> &mut Self::Output {
        let size = size_of::<T>() as u64;
        let offset = index * size;
        let bytes = unsafe { self.inner.get_memory(offset..offset + size) };
        unsafe { &mut *((bytes.as_ptr()) as *mut T) }
    }
}

impl<T> Index<Range<u64>> for Buffer<T> {
    type Output = [T];
    #[inline]
    fn index(&self, range: Range<u64>) -> &Self::Output {
        let bytes = unsafe {
            self.inner
                .get_memory(range.start * size_of::<T>() as u64..range.end * size_of::<T>() as u64)
        };
        let len = (range.end - range.start) as usize;
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, len) }
    }
}

impl<T> IndexMut<Range<u64>> for Buffer<T> {
    #[inline]
    fn index_mut(&mut self, range: Range<u64>) -> &mut Self::Output {
        let bytes = unsafe {
            self.inner
                .get_memory(range.start * size_of::<T>() as u64..range.end * size_of::<T>() as u64)
        };
        let len = (range.end - range.start) as usize;
        unsafe { std::slice::from_raw_parts_mut(bytes.as_ptr() as *mut T, len) }
    }
}

impl<T> Index<RangeFull> for Buffer<T> {
    type Output = [T];
    #[inline]
    fn index(&self, _index: RangeFull) -> &Self::Output {
        &self[0..self.len()]
    }
}

impl<T> IndexMut<RangeFull> for Buffer<T> {
    #[inline]
    fn index_mut(&mut self, _index: RangeFull) -> &mut Self::Output {
        let size = self.len();
        &mut self[0..size]
    }
}

impl<T> Index<RangeTo<u64>> for Buffer<T> {
    type Output = [T];
    #[inline]
    fn index(&self, range: RangeTo<u64>) -> &Self::Output {
        &self[0..range.end]
    }
}

impl<T> IndexMut<RangeTo<u64>> for Buffer<T> {
    #[inline]
    fn index_mut(&mut self, range: RangeTo<u64>) -> &mut Self::Output {
        &mut self[0..range.end]
    }
}

impl<T> Index<RangeFrom<u64>> for Buffer<T> {
    type Output = [T];
    #[inline]
    fn index(&self, range: RangeFrom<u64>) -> &Self::Output {
        &self[range.start..self.len()]
    }
}

impl<T> IndexMut<RangeFrom<u64>> for Buffer<T> {
    #[inline]
    fn index_mut(&mut self, range: RangeFrom<u64>) -> &mut Self::Output {
        let size = self.len();
        &mut self[range.start..size]
    }
}
