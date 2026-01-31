use std::{error::Error, sync::Arc};

use ash::vk;

use crate::device::Device;

pub(crate) struct RenderPassInfo {
    pub samples: vk::SampleCountFlags,
    pub format: vk::Format,
    pub present: bool,
}

pub(crate) struct RenderPass {
    pub handle: vk::RenderPass,
    pub info: RenderPassInfo,
    pub device: Arc<Device>,
}

unsafe impl Send for RenderPass {}
unsafe impl Sync for RenderPass {}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_render_pass(self.handle, None) }
    }
}

impl RenderPass {
    pub fn new(device: Arc<Device>, info: RenderPassInfo) -> Result<Arc<Self>, Box<dyn Error>> {
        let initial_layout = vk::ImageLayout::UNDEFINED;
        let final_layout = if info.present {
            vk::ImageLayout::PRESENT_SRC_KHR
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };

        let color_attachment = vk::AttachmentDescription::default()
            .format(info.format)
            .samples(info.samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(initial_layout)
            .final_layout(if info.samples != vk::SampleCountFlags::TYPE_1 {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            } else {
                final_layout
            });

        let depth_format = match device.physical_device.find_depth_format(
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        ) {
            Some(format) => format,
            None => {
                return Err("cannot find supported depth format".into());
            }
        };

        let depth_attachment = vk::AttachmentDescription::default()
            .format(depth_format)
            .samples(info.samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(initial_layout)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_attachment_resolve = vk::AttachmentDescription::default()
            .format(info.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(initial_layout)
            .final_layout(final_layout);

        let mut attachments = vec![color_attachment, depth_attachment];

        if info.samples != vk::SampleCountFlags::TYPE_1 {
            attachments.push(color_attachment_resolve);
        }

        let color_attachment_reference = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_attachment_reference = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_attachments = &[color_attachment_reference];

        let color_attachment_resolve_reference = vk::AttachmentReference::default()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let mut resolve_attachments = vec![];

        if info.samples != vk::SampleCountFlags::TYPE_1 {
            resolve_attachments.push(color_attachment_resolve_reference)
        }

        let mut subpass = vk::SubpassDescription::default()
            .color_attachments(color_attachments)
            .depth_stencil_attachment(&depth_attachment_reference)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

        if info.samples != vk::SampleCountFlags::TYPE_1 {
            subpass = subpass.resolve_attachments(&resolve_attachments)
        }

        let subpasses = &[subpass];

        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::TOP_OF_PIPE)
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );

        let end_dependency = vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::empty());

        let dependencies = &[dependency, end_dependency];

        let render_pass_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(subpasses)
            .dependencies(dependencies);

        let render_pass = unsafe {
            device
                .handle
                .create_render_pass(&render_pass_create_info, None)
        }?;

        Ok(Arc::new(Self {
            handle: render_pass,
            info,
            device,
        }))
    }
}
