use std::{error::Error, marker::PhantomData, sync::Arc};

use ash::vk;

use crate::{
    error,
    errors::PipelineError,
    pipeline::{Pipeline, PipelineInfo, descriptor::layout::PipelineLayout, shader::Shader},
};

impl Pipeline<u32> {
    pub fn new_compute(
        pipeline_layout: Arc<PipelineLayout>,
        shader: Arc<Shader>,
    ) -> Result<Arc<Pipeline<u32>>, Box<dyn Error>> {
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(shader.stage)
            .module(shader.handle)
            .name(&shader.entry_point);

        let create_info = vk::ComputePipelineCreateInfo::default()
            .layout(pipeline_layout.handle)
            .stage(stage);

        let pipeline = match unsafe {
            pipeline_layout.device.handle.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[create_info],
                None,
            )
        } {
            Ok(pipelines) => pipelines[0],
            Err(e) => {
                return error!(PipelineError, "cannot create compute pipeline: {e:?}");
            }
        };

        Ok(Arc::new(Self {
            handle: pipeline,
            pipeline_layout,
            bind_point: vk::PipelineBindPoint::COMPUTE,
            _info: PipelineInfo::None,
            _shaders: vec![shader],
            _render_target: None,
            _tp: PhantomData,
        }))
    }
}
