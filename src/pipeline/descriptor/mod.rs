pub mod descriptor_set_layout;
pub mod layout;

use std::{
    error::Error,
    sync::{Arc, Mutex},
};

use ash::vk;

use crate::device::Device;

pub struct DescriptorPool {
    pub(crate) handle: vk::DescriptorPool,
    pub device: Arc<Device>,
}

unsafe impl Send for DescriptorPool {}
unsafe impl Sync for DescriptorPool {}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle
                .destroy_descriptor_pool(self.handle, None);
        }
    }
}

impl DescriptorPool {
    pub fn new(
        device: Arc<Device>,
        descriptor_sizes: &[vk::DescriptorPoolSize],
    ) -> Result<Arc<Mutex<Self>>, Box<dyn Error>> {
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(descriptor_sizes)
            .max_sets(
                descriptor_sizes
                    .iter()
                    .map(|size| size.descriptor_count)
                    .sum(),
            )
            .flags(
                vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND
                    & vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
            );

        let descriptor_pool = unsafe { device.handle.create_descriptor_pool(&pool_info, None) }?;

        let descriptor_pool = Arc::new(Mutex::new(Self {
            handle: descriptor_pool,
            device,
        }));

        Ok(descriptor_pool)
    }
}
