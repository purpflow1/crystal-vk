use std::{error::Error, sync::Arc};

use ash::vk;

use crate::{
    device::Device, error, errors::ImageError, image::Image,
    pipeline::descriptor::descriptor_set_layout::binding::Binding,
};

pub struct SamplerInfo {
    pub filter: vk::Filter,
    pub address_mode: vk::SamplerAddressMode,
    pub anisotropy_texels: f32,
    pub max_lod: f32,
}

pub struct Sampler {
    pub(crate) handle: vk::Sampler,
    pub device: Arc<Device>,
}

impl Binding for Sampler {}

impl Binding for (Arc<Image>, Arc<Sampler>) {}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_sampler(self.handle, None);
        }
    }
}

impl Sampler {
    pub fn new(device: Arc<Device>, info: SamplerInfo) -> Result<Arc<Self>, Box<dyn Error>> {
        let anisotropy_enable = info.anisotropy_texels != 1.;

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(info.filter)
            .min_filter(info.filter)
            .address_mode_u(info.address_mode)
            .address_mode_v(info.address_mode)
            .address_mode_w(info.address_mode)
            .anisotropy_enable(anisotropy_enable)
            .max_anisotropy(info.anisotropy_texels)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.)
            .min_lod(0.)
            .max_lod(info.max_lod);

        match unsafe { device.handle.create_sampler(&sampler_info, None) } {
            Ok(sampler) => Ok(Arc::new(Self {
                handle: sampler,
                device,
            })),
            Err(e) => {
                error!(ImageError, "cannot create sampler: {e}")
            }
        }
    }
}
