use std::{error::Error, sync::Arc};

use ash::vk;

use crate::pipeline::{Pipeline, PipelineInfo, descriptor::layout::PipelineLayout, shader::Shader};

impl Pipeline {
    pub fn new_compute(
        pipeline_layout: Arc<PipelineLayout>,
        shader: Arc<Shader>,
        cache: Option<&[u8]>,
    ) -> Result<Arc<Pipeline>, Box<dyn Error>> {
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(shader.stage)
            .module(shader.handle)
            .name(&shader.entry_point);

        let create_info = vk::ComputePipelineCreateInfo::default()
            .layout(pipeline_layout.handle)
            .stage(stage);

        let mut cache_create_info = vk::PipelineCacheCreateInfo::default();
        if let Some(cache) = cache {
            cache_create_info = cache_create_info.initial_data(cache);
        }

        let cache = unsafe {
            pipeline_layout
                .device
                .handle
                .create_pipeline_cache(&cache_create_info, None)?
        };

        let pipeline = match unsafe {
            pipeline_layout
                .device
                .handle
                .create_compute_pipelines(cache, &[create_info], None)
        } {
            Ok(pipelines) => pipelines[0],
            Err(e) => {
                return Err(e.1.into());
            }
        };

        Ok(Arc::new(Self {
            handle: pipeline,
            pipeline_layout,
            bind_point: vk::PipelineBindPoint::COMPUTE,
            _info: PipelineInfo::None,
            _shaders: vec![shader],
            _render_target: None,
            cache,
        }))
    }
}
