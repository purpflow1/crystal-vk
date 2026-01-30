pub mod descriptor_set;

use std::{collections::BTreeMap, error::Error, sync::Arc};

use ash::vk;

use crate::{device::Device, error, errors::DescriptorError};

#[derive(Clone)]
pub struct LayoutAllocInfo {
    pub stages: vk::ShaderStageFlags,
    pub typ: vk::DescriptorType,
    pub count: u32,
}

pub struct DescriptorSetLayout {
    pub handle: vk::DescriptorSetLayout,
    pub alloc_info: BTreeMap<u32, LayoutAllocInfo>,

    pub device: Arc<Device>,
}

unsafe impl Send for DescriptorSetLayout {}
unsafe impl Sync for DescriptorSetLayout {}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle
                .destroy_descriptor_set_layout(self.handle, None)
        }
    }
}

impl DescriptorSetLayout {
    pub fn new(
        device: Arc<Device>,
        layout_alloc_info: BTreeMap<u32, LayoutAllocInfo>,
    ) -> Result<Arc<Self>, Box<dyn Error>> {
        let bindings = layout_alloc_info
            .iter()
            .map(|(binding, layout_info)| {
                vk::DescriptorSetLayoutBinding::default()
                    .descriptor_type(layout_info.typ)
                    .stage_flags(layout_info.stages)
                    .descriptor_count(layout_info.count)
                    .binding(*binding)
            })
            .collect::<Vec<_>>();

        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let handle = match unsafe {
            device
                .handle
                .create_descriptor_set_layout(&create_info, None)
        } {
            Ok(handle) => handle,
            Err(e) => return error!(DescriptorError, "cannot create descriptor set layout: {e}"),
        };

        Ok(Arc::new(Self {
            handle,
            alloc_info: layout_alloc_info,
            device,
        }))
    }
}
