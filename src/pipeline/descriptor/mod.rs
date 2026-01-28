pub mod descriptor_set_layout;
pub mod layout;

use std::{
    error::Error,
    sync::{Arc, Mutex},
};

use ash::vk;

use crate::{device::Device, error, errors::DeviceError};

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
    pub fn new(device: Arc<Device>) -> Result<Arc<Mutex<Self>>, Box<dyn Error>> {
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .descriptor_count(64)
                .ty(vk::DescriptorType::UNIFORM_BUFFER),
            vk::DescriptorPoolSize::default()
                .descriptor_count(64)
                .ty(vk::DescriptorType::STORAGE_BUFFER),
            vk::DescriptorPoolSize::default()
                .descriptor_count(64)
                .ty(vk::DescriptorType::SAMPLED_IMAGE),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(64 * 3)
            .flags(
                vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND
                    & vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
            );

        let descriptor_pool =
            match unsafe { device.handle.create_descriptor_pool(&pool_info, None) } {
                Ok(descriptor_pool) => descriptor_pool,
                Err(e) => {
                    return error!(DeviceError, "cannot create descriptor pool: {e}");
                }
            };

        let descriptor_pool = Arc::new(Mutex::new(Self {
            handle: descriptor_pool,
            device,
        }));

        Ok(descriptor_pool)
    }
}
