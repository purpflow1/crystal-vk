use std::{
    collections::BTreeMap,
    error::Error,
    sync::{Arc, Mutex, RwLock},
};

use ash::vk;

use crate::{
    buffer::Buffer,
    error,
    errors::DescriptorError,
    image::{Image, sampler::Sampler},
    pipeline::descriptor::{
        DescriptorPool,
        descriptor_set_layout::{DescriptorSetLayout, binding::Binding},
    },
};

pub struct DescriptorSet {
    pub(crate) handle: vk::DescriptorSet,
    pub(crate) descriptor_set_layout: Arc<DescriptorSetLayout>,
    descriptor_pool: Arc<Mutex<DescriptorPool>>,

    bindings: BTreeMap<u32, Arc<dyn Binding>>,
}

impl DescriptorSet {
    pub fn bind_combined_image_sampler(
        &mut self,
        image: Arc<Image>,
        sampler: Arc<Sampler>,
        binding: u32,
        array_offset: u32,
        array_count: u32,
    ) -> Result<(), Box<dyn Error>> {
        let device = image.device.clone();

        let image_infos = [vk::DescriptorImageInfo::default()
            .image_view(image.image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(sampler.handle)];

        let typ = if let Some(alloc_info) = self.descriptor_set_layout.alloc_info.get(&binding) {
            alloc_info.typ
        } else {
            return error!(DescriptorError, "no such binding");
        };

        let descriptor_write = vk::WriteDescriptorSet::default()
            .descriptor_type(typ)
            .dst_set(self.handle)
            .dst_binding(binding)
            .dst_array_element(array_offset)
            .descriptor_count(array_count)
            .image_info(&image_infos);

        unsafe {
            device
                .handle
                .update_descriptor_sets(&[descriptor_write], &[])
        };

        self.bindings.insert(binding, Arc::new((image, sampler)));

        Ok(())
    }

    pub fn bind_buffer<T: 'static>(
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

        self.bindings.insert(binding, buffer);

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
            bindings: BTreeMap::new(),
        })))
    }
}
