use std::{
    error::Error,
    sync::{Arc, RwLock},
};

use ash::vk;

use crate::{error, errors::DeviceError, image::Image, render::render_pass::RenderPass};

pub(crate) struct FramebufferPool {
    handles: Vec<vk::Framebuffer>,
    pub(crate) attachments: Vec<Arc<Image>>,
    pub render_pass: Arc<RenderPass>,
}

unsafe impl Send for FramebufferPool {}
unsafe impl Sync for FramebufferPool {}

impl Drop for FramebufferPool {
    fn drop(&mut self) {
        self.handles.iter().for_each(|handle| unsafe {
            self.render_pass
                .device
                .handle
                .destroy_framebuffer(*handle, None)
        })
    }
}

impl FramebufferPool {
    pub(crate) fn get_framebuffer(&self, image_index: u32) -> vk::Framebuffer {
        self.handles[image_index as usize]
    }

    pub fn new(
        render_pass: Arc<RenderPass>,
        images: Vec<Arc<Image>>,
    ) -> Result<Arc<RwLock<Self>>, Box<dyn Error>> {
        let mut attachments_child = Vec::with_capacity(256);
        let mut framebuffers = Vec::with_capacity(256);

        for image in images {
            let depth_image = Image::new_framebuffer_depth(
                render_pass.device.clone(),
                image.info.extent,
                render_pass.info.samples,
            )?;

            attachments_child.push(depth_image.clone());
            attachments_child.push(image.clone());

            let attachments = if render_pass.info.samples != vk::SampleCountFlags::TYPE_1 {
                let color_image = Image::new_framebuffer_color(
                    render_pass.device.clone(),
                    image.info.extent,
                    image.info.format,
                    render_pass.info.samples,
                )?;

                attachments_child.push(color_image.clone());

                vec![
                    color_image.image_view,
                    depth_image.image_view,
                    image.image_view,
                ]
            } else {
                vec![image.image_view, depth_image.image_view]
            };

            let create_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass.handle)
                .attachments(&attachments)
                .width(image.info.extent[0])
                .height(image.info.extent[1])
                .layers(1);

            match unsafe {
                render_pass
                    .device
                    .handle
                    .create_framebuffer(&create_info, None)
            } {
                Ok(framebuffer) => framebuffers.push(framebuffer),
                Err(e) => return error!(DeviceError, "cannot create framebuffer: {e}"),
            }
        }

        Ok(Arc::new(RwLock::new(FramebufferPool {
            handles: framebuffers,
            attachments: attachments_child,
            render_pass,
        })))
    }
}
