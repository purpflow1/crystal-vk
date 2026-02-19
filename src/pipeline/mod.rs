pub mod compute;
pub mod graphics;

use graphics::*;

use std::{error::Error, sync::Arc};

use ash::vk;

use crate::{
    pipeline::{descriptor::layout::PipelineLayout, shader::Shader},
    render::RenderTarget,
    traits::CommandBufferBinding,
};

pub mod attribute;
pub mod descriptor;
pub mod shader;

pub enum PipelineInfo {
    Graphics(GraphicsPipelineInfo),
    None,
}

pub struct Pipeline {
    pub(crate) handle: vk::Pipeline,
    pub pipeline_layout: Arc<PipelineLayout>,
    pub bind_point: vk::PipelineBindPoint,
    pub(crate) _info: PipelineInfo,
    _shaders: Vec<Arc<Shader>>,
    _render_target: Option<Arc<RenderTarget>>,
    cache: vk::PipelineCache,
}

unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}
impl CommandBufferBinding for Pipeline {}

impl Drop for Pipeline {
    fn drop(&mut self) {
        let device = self.pipeline_layout.device.clone();

        unsafe {
            device.handle.destroy_pipeline_cache(self.cache, None);
            device.handle.destroy_pipeline(self.handle, None);
        }
    }
}

impl Pipeline {
    pub fn cache(&self) -> Result<Vec<u8>, Box<dyn Error>> {
        let device = self.pipeline_layout.device.clone();
        let data = unsafe { device.handle.get_pipeline_cache_data(self.cache)? };
        Ok(data)
    }
}
