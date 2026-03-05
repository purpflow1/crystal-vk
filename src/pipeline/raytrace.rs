use std::{error::Error, sync::Arc};

use ash::vk;

use crate::pipeline::{Pipeline, PipelineInfo, descriptor::layout::PipelineLayout, shader::Shader};

impl Pipeline {
    /// Creates a ray‑tracing pipeline with automatic shader‑group configuration.
    ///
    /// The function inspects each supplied `Shader`'s `stage` flag and builds the appropriate
    /// `vk::RayTracingShaderGroupCreateInfoKHR` entries:
    ///   * Ray‑generation, miss and callable shaders → **GENERAL** groups.
    ///   * Closest‑hit (plus optional any‑hit) shaders → **TRIANGLES_HIT_GROUP**.
    ///   * Intersection shaders (procedural geometry) → **PROCEDURAL_HIT_GROUP**.
    ///
    /// If no shaders are supplied the function returns an error.
    pub fn new_raytrace(
        pipeline_layout: Arc<PipelineLayout>,
        shaders: Vec<Arc<Shader>>,
        cache: Option<&[u8]>,
    ) -> Result<Arc<Pipeline>, Box<dyn Error>> {
        if shaders.is_empty() {
            return Err("no shaders specified".into());
        }

        let mut stages = Vec::with_capacity(shaders.len());

        let mut raygen = Vec::new();
        let mut miss = Vec::new();
        let mut callable = Vec::new();
        let mut closest_hit = Vec::new();
        let mut any_hit = Vec::new();
        let mut intersection = Vec::new();

        for (idx, shader) in shaders.iter().enumerate() {
            stages.push(vk::PipelineShaderStageCreateInfo {
                stage: shader.stage,
                module: shader.handle,
                p_name: shader.entry_point.as_ptr(),
                ..Default::default()
            });

            let idx_u32 = idx as u32;
            if shader.stage.contains(vk::ShaderStageFlags::RAYGEN_KHR) {
                raygen.push(idx_u32);
            } else if shader.stage.contains(vk::ShaderStageFlags::MISS_KHR) {
                miss.push(idx_u32);
            } else if shader.stage.contains(vk::ShaderStageFlags::CALLABLE_KHR) {
                callable.push(idx_u32);
            } else if shader.stage.contains(vk::ShaderStageFlags::CLOSEST_HIT_KHR) {
                closest_hit.push(idx_u32);
            } else if shader.stage.contains(vk::ShaderStageFlags::ANY_HIT_KHR) {
                any_hit.push(idx_u32);
            } else if shader
                .stage
                .contains(vk::ShaderStageFlags::INTERSECTION_KHR)
            {
                intersection.push(idx_u32);
            }
        }

        let mut groups = Vec::new();

        // Ray‑generation groups (GENERAL)
        for &idx in &raygen {
            groups.push(
                vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                    .general_shader(idx)
                    .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                    .any_hit_shader(vk::SHADER_UNUSED_KHR)
                    .intersection_shader(vk::SHADER_UNUSED_KHR),
            );
        }

        // Miss groups (GENERAL)
        for &idx in &miss {
            groups.push(
                vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                    .general_shader(idx)
                    .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                    .any_hit_shader(vk::SHADER_UNUSED_KHR)
                    .intersection_shader(vk::SHADER_UNUSED_KHR),
            );
        }

        // Callable groups (GENERAL)
        for &idx in &callable {
            groups.push(
                vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                    .general_shader(idx)
                    .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                    .any_hit_shader(vk::SHADER_UNUSED_KHR)
                    .intersection_shader(vk::SHADER_UNUSED_KHR),
            );
        }

        // Triangle hit groups – pair each closest‑hit shader with an any‑hit shader
        // if one is available.
        let mut any_hit_iter = any_hit.into_iter();
        for &ch_idx in &closest_hit {
            let any_idx = any_hit_iter.next().unwrap_or(vk::SHADER_UNUSED_KHR);
            groups.push(
                vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                    .general_shader(vk::SHADER_UNUSED_KHR)
                    .closest_hit_shader(ch_idx)
                    .any_hit_shader(any_idx)
                    .intersection_shader(vk::SHADER_UNUSED_KHR),
            );
        }

        // Procedural hit groups – use INTERSECTION shaders (and optional any‑hit).
        for &is_idx in &intersection {
            let any_idx = any_hit_iter.next().unwrap_or(vk::SHADER_UNUSED_KHR);
            groups.push(
                vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
                    .general_shader(vk::SHADER_UNUSED_KHR)
                    .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                    .any_hit_shader(any_idx)
                    .intersection_shader(is_idx),
            );
        }

        let create_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .layout(pipeline_layout.handle)
            .stages(&stages)
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(1);

        let mut cache_ci = vk::PipelineCacheCreateInfo::default();
        if let Some(data) = cache {
            cache_ci = cache_ci.initial_data(data);
        }

        let pipeline_cache = unsafe {
            pipeline_layout
                .device
                .handle
                .create_pipeline_cache(&cache_ci, None)?
        };

        let rt_loader = ash::khr::ray_tracing_pipeline::Device::new(
            &pipeline_layout.device.instance.handle,
            &pipeline_layout.device.handle,
        );

        let pipeline = unsafe {
            rt_loader
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    pipeline_cache,
                    &[create_info],
                    None,
                )
                .map_err(|(_, err)| err)?[0]
        };

        Ok(Arc::new(Self {
            handle: pipeline,
            pipeline_layout,
            bind_point: vk::PipelineBindPoint::RAY_TRACING_KHR,
            _info: PipelineInfo::None,
            _shaders: shaders,
            _render_target: None,
            cache: pipeline_cache,
        }))
    }
}
