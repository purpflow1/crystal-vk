pub mod compute;
pub mod graphics;

use graphics::*;

use std::{marker::PhantomData, sync::Arc};

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

pub struct Pipeline<V> {
    pub(crate) handle: vk::Pipeline,
    pub pipeline_layout: Arc<PipelineLayout>,
    pub bind_point: vk::PipelineBindPoint,
    pub(crate) _info: PipelineInfo,
    _shaders: Vec<Arc<Shader>>,
    _render_target: Option<Arc<RenderTarget>>,
    _tp: PhantomData<V>,
}

unsafe impl<T> Send for Pipeline<T> {}
unsafe impl<T> Sync for Pipeline<T> {}
impl<T> CommandBufferBinding for Pipeline<T> {}

impl<V> Drop for Pipeline<V> {
    fn drop(&mut self) {
        unsafe {
            self.pipeline_layout
                .device
                .handle
                .destroy_pipeline(self.handle, None);
        }
    }
}
