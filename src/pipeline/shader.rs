use std::{error::Error, ffi::CString, sync::Arc};

use ash::vk;

use crate::{device::Device, error, errors::PipelineError};

pub struct Shader {
    pub(crate) handle: vk::ShaderModule,
    pub(crate) entry_point: CString,
    pub(crate) stage: vk::ShaderStageFlags,
    pub device: Arc<Device>,
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_shader_module(self.handle, None) }
    }
}

impl Shader {
    pub fn new(
        device: Arc<Device>,
        entry_point: CString,
        stage: vk::ShaderStageFlags,
        code: Vec<u32>,
    ) -> Result<Arc<Self>, Box<dyn Error>> {
        let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(&code);

        let module = match unsafe {
            device
                .handle
                .create_shader_module(&shader_module_create_info, None)
        } {
            Ok(module) => module,
            Err(e) => {
                return error!(PipelineError, "cannot create shader module: {e}");
            }
        };

        Ok(Arc::new(Self {
            handle: module,
            entry_point,
            stage,
            device,
        }))
    }
}
