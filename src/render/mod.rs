pub mod framebuffer;
mod render_pass;
pub mod surface;
pub mod swapchain;

use std::{
    error::Error,
    sync::{Arc, RwLock},
};

use ash::vk;

use crate::{
    device::Device,
    image::{Image, ImageType},
    render::{
        framebuffer::FramebufferPool,
        render_pass::{RenderPass, RenderPassInfo},
    },
    traits::CommandBufferBinding,
};

pub struct RenderTarget {
    pub(crate) framebuffer: Arc<RwLock<FramebufferPool>>,
    pub device: Arc<Device>,
}

unsafe impl Send for RenderTarget {}
unsafe impl Sync for RenderTarget {}

impl CommandBufferBinding for RenderTarget {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl RenderTarget {
    pub fn new(
        device: Arc<Device>,
        image_sequence: Vec<Arc<Image>>,
        msaa_samples: u8,
    ) -> Result<Arc<Self>, Box<dyn Error>> {
        let counts = device
            .physical_device
            .info
            .properties
            .limits
            .framebuffer_color_sample_counts
            & device
                .physical_device
                .info
                .properties
                .limits
                .framebuffer_depth_sample_counts;

        let samples = match msaa_samples {
            2 => vk::SampleCountFlags::TYPE_2,
            4 => vk::SampleCountFlags::TYPE_4,
            8 => vk::SampleCountFlags::TYPE_8,
            16 => vk::SampleCountFlags::TYPE_16,
            32 => vk::SampleCountFlags::TYPE_32,
            64 => vk::SampleCountFlags::TYPE_64,
            _ => vk::SampleCountFlags::TYPE_1,
        };

        if !samples.intersects(counts) {
            return Err(
                format!("device is not supported for sample count: {}", msaa_samples).into(),
            );
        };

        let present = image_sequence[0].info.typ == ImageType::Swapchain;

        let render_pass = RenderPass::new(
            device.clone(),
            RenderPassInfo {
                samples,
                format: image_sequence[0].info.format,
                present,
            },
        )?;

        let framebuffer = FramebufferPool::new(render_pass.clone(), image_sequence)?;

        Ok(Arc::new(Self {
            framebuffer,
            device,
        }))
    }
}
