use std::{error::Error, iter::zip, sync::Arc};

use ash::vk;

use crate::{
    pipeline::{
        Pipeline, PipelineInfo, attribute::Attribute, descriptor::layout::PipelineLayout,
        shader::Shader,
    },
    render::RenderTarget,
};

pub struct GraphicsPipelineInfo {
    pub viewports: Vec<vk::Viewport>,
    pub scissors: Vec<vk::Rect2D>,
    pub line_width: f32,
    pub polygon_mode: vk::PolygonMode,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub vertex_attributes: Vec<Attribute>,
}

impl Default for GraphicsPipelineInfo {
    fn default() -> Self {
        Self {
            viewports: Vec::with_capacity(0),
            scissors: Vec::with_capacity(0),
            line_width: 1.,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            vertex_attributes: vec![],
        }
    }
}

impl Pipeline {
    pub fn new_graphics(
        pipeline_layout: Arc<PipelineLayout>,
        render_target: Arc<RenderTarget>,
        shaders: Vec<Arc<Shader>>,
        mut pipeline_info: GraphicsPipelineInfo,
        cache: Option<&[u8]>,
    ) -> Result<Arc<Pipeline>, Box<dyn Error>> {
        if shaders.is_empty() {
            return Err("no shaders specified".into());
        }

        let mut stages = Vec::new();

        for shader in shaders.clone() {
            let stage = vk::PipelineShaderStageCreateInfo {
                stage: shader.stage,
                module: shader.handle,
                p_name: shader.entry_point.as_ptr(),
                ..Default::default()
            };

            stages.push(stage);
        }

        let attributes = pipeline_info.vertex_attributes.clone();

        let binding_descriptions = &[vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(attributes.iter().map(|a| a.size as u32).sum())
            .input_rate(vk::VertexInputRate::VERTEX)];

        let mut attribute_descriptions = vec![];

        for (location, attribute) in zip(0..attributes.len() as u32, attributes) {
            let attribute_description = vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(location)
                .format(match attribute.size {
                    4 => vk::Format::R32_SFLOAT,
                    8 => vk::Format::R32G32_SFLOAT,
                    12 => vk::Format::R32G32B32_SFLOAT,
                    16 => vk::Format::R32G32B32A32_SFLOAT,
                    _ => vk::Format::R32G32B32_SFLOAT,
                })
                .offset(attribute.offset as u32);
            attribute_descriptions.push(attribute_description);
        }

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let dynamic_states = &[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(dynamic_states);

        let target = render_target.clone();
        let framebuffer_lock = target.framebuffer.read().unwrap();

        let (viewports, scissors) =
            if pipeline_info.viewports.is_empty() || pipeline_info.scissors.is_empty() {
                let extent = framebuffer_lock.attachments[0].info.extent;
                (
                    vec![
                        vk::Viewport::default()
                            .width(extent[0] as f32)
                            .height(extent[1] as f32),
                    ],
                    vec![vk::Rect2D::default().extent(vk::Extent2D {
                        width: extent[0],
                        height: extent[1],
                    })],
                )
            } else {
                (
                    pipeline_info.viewports.clone(),
                    pipeline_info.scissors.clone(),
                )
            };

        pipeline_info.viewports = viewports;
        pipeline_info.scissors = scissors;

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&pipeline_info.viewports)
            .scissors(&pipeline_info.scissors);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(pipeline_info.polygon_mode)
            .line_width(pipeline_info.line_width)
            .cull_mode(pipeline_info.cull_mode)
            .front_face(pipeline_info.front_face)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(framebuffer_lock.render_pass.info.samples);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let attachments = &[color_blend_attachment];

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(attachments);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false);

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .depth_stencil_state(&depth_stencil_state)
            .layout(pipeline_layout.handle)
            .render_pass(framebuffer_lock.render_pass.handle);

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

        match unsafe {
            render_target.device.handle.create_graphics_pipelines(
                cache,
                &[pipeline_create_info],
                None,
            )
        } {
            Ok(pipelines) => Ok(Arc::new(Self {
                handle: pipelines[0],
                _info: PipelineInfo::Graphics(pipeline_info),
                bind_point: vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                _shaders: shaders,
                _render_target: Some(render_target),
                cache,
            })),
            Err(e) => Err(e.1.into()),
        }
    }
}
