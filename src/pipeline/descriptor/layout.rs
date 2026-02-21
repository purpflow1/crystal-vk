use std::{
    error::Error,
    sync::{Arc, Mutex},
};

use ash::vk;

use crate::{
    device::Device,
    pipeline::descriptor::{DescriptorPool, descriptor_set_layout::DescriptorSetLayout},
};

pub struct PipelineLayout {
    pub(crate) handle: vk::PipelineLayout,
    pub(crate) _descriptor_set_layouts: Vec<Arc<DescriptorSetLayout>>,
    pub device: Arc<Device>,
    pub descriptor_pool: Arc<Mutex<DescriptorPool>>,
}

unsafe impl Send for PipelineLayout {}
unsafe impl Sync for PipelineLayout {}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
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

        let pipeline_layout = unsafe { device.handle.create_pipeline_layout(&create_info, None) }?;

        drop(lock);

        Ok(Arc::new(Self {
            handle: pipeline_layout,
            _descriptor_set_layouts: descriptor_set_layouts,
            device,
            descriptor_pool,
        }))
    }
}
