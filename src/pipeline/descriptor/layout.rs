use std::{
    cell::RefCell,
    collections::BTreeMap,
    error::Error,
    iter::zip,
    sync::{Arc, Mutex, RwLock},
};

use ash::vk;

use crate::{
    buffer::Buffer,
    device::Device,
    error,
    errors::DescriptorError,
    pipeline::descriptor::{DescriptorPool, descriptor_set_layout::DescriptorSetLayout},
};

pub struct PipelineLayout {
    pub(crate) handle: vk::PipelineLayout,
    pub(crate) descriptor_set_layouts: Vec<Arc<DescriptorSetLayout>>,
    pub device: Arc<Device>,
    pub descriptor_pool: Arc<Mutex<DescriptorPool>>,
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        let lock = self.descriptor_pool.lock().unwrap();
        unsafe {
            self.device
                .handle
                .destroy_pipeline_layout(self.handle, None);
        }
    }
}

impl PipelineLayout {
    pub fn new(
        descriptor_pool: Arc<Mutex<DescriptorPool>>,
        descriptor_set_layouts: Vec<Arc<DescriptorSetLayout>>,
    ) -> Result<Arc<Self>, Box<dyn Error>> {
        let descriptor_set_layouts_raw = descriptor_set_layouts
            .iter()
            .map(|descriptor_set_layout| descriptor_set_layout.handle)
            .collect::<Vec<_>>();

        let lock = descriptor_pool.lock().unwrap();

        let create_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts_raw);

        let device = lock.device.clone();

        let pipeline_layout =
            match unsafe { device.handle.create_pipeline_layout(&create_info, None) } {
                Ok(layout) => layout,
                Err(e) => {
                    return error!(DescriptorError, "cannot create pipeline layout: {e}");
                }
            };

        drop(lock);

        Ok(Arc::new(Self {
            handle: pipeline_layout,
            descriptor_set_layouts,
            device,
            descriptor_pool,
        }))
    }
}
