use std::{
    error::Error,
    sync::{Arc, Mutex, RwLock},
};

use ash::vk;

use crate::{
    buffer::Buffer,
    error,
    errors::DescriptorError,
    pipeline::descriptor::{DescriptorPool, descriptor_set_layout::DescriptorSetLayout},
};

enum Bound<T> {
    None,
    Buffer(Arc<RwLock<Buffer<T>>>),
}

pub struct DescriptorSet<T> {
    pub(crate) handle: vk::DescriptorSet,
    pub(crate) descriptor_set_layout: Arc<DescriptorSetLayout>,
    descriptor_pool: Arc<Mutex<DescriptorPool>>,

    bound: Bound<T>,
}

impl<T> DescriptorSet<T> {
    pub fn bind_buffer(
        &mut self,
        buffer: Arc<RwLock<Buffer<T>>>,
        binding: u32,
        array_offset: u32,
        array_count: u32,
    ) -> Result<(), Box<dyn Error>> {
        let pool = self.descriptor_pool.clone();
        let pool_lock = pool.lock().unwrap();
        let device = pool_lock.device.clone();
        let buffer_lock = buffer.write().unwrap();

        let typ = if let Some(alloc_info) = self.descriptor_set_layout.alloc_info.get(&binding) {
            alloc_info.typ
        } else {
            return error!(DescriptorError, "no such binding");
        };

        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer_lock.as_raw())
            .range(buffer_lock.len() * size_of::<T>() as u64)];

        let descriptor_write = vk::WriteDescriptorSet::default()
            .dst_set(self.handle)
            .dst_binding(binding)
            .dst_array_element(array_offset)
            .descriptor_type(typ)
            .descriptor_count(array_count)
            .buffer_info(&buffer_info);

        unsafe {
            device
                .handle
                .update_descriptor_sets(&[descriptor_write], &[])
        };

        drop(buffer_lock);

        self.bound = Bound::Buffer(buffer);
        Ok(())
    }

    pub fn new(
        descriptor_pool: Arc<Mutex<DescriptorPool>>,
        descriptor_set_layout: Arc<DescriptorSetLayout>,
    ) -> Result<Arc<Mutex<Self>>, Box<dyn Error>> {
        let lock = descriptor_pool.lock().unwrap();

        let set_layout = [descriptor_set_layout.handle];

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(lock.handle)
            .set_layouts(&set_layout);

        let descriptor_set =
            match unsafe { lock.device.handle.allocate_descriptor_sets(&alloc_info) } {
                Ok(descriptor_sets) => descriptor_sets[0],
                Err(e) => {
                    return error!(DescriptorError, "cannot allocate descriptor sets: {e}");
                }
            };

        drop(lock);

        Ok(Arc::new(Mutex::new(Self {
            handle: descriptor_set,
            descriptor_set_layout: descriptor_set_layout,
            descriptor_pool,
            bound: Bound::None,
        })))
    }
}
