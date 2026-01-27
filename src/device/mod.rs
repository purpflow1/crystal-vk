pub mod physical_device;
pub mod queue;

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{error::Error, sync::Arc};

use crate::{
    device::physical_device::PhysicalDevice,
    errors::DeviceError,
    instance,
    render::surface::{self, window},
};

pub struct Device {
    pub(crate) handle: ash::Device,
    pub(crate) instance: Arc<instance::Instance>,
    pub(crate) surface: Arc<surface::Surface>,
    pub(crate) physical_device: Arc<physical_device::PhysicalDevice>,
    pub(crate) extensions: Vec<String>,
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { self.handle.destroy_device(None) }
    }
}

impl Device {
    pub fn compute() -> Result<Arc<Self>, Box<dyn Error>> {
        //Self::new_in::<NullWindow>(None)
        todo!()
    }

    pub fn with_present<T: HasWindowHandle + HasDisplayHandle>(
        window: &T,
    ) -> Result<Arc<Self>, Box<dyn Error>> {
        let ws_handlers = window::WindowSystemRawHandlers::new(window)?;

        let instance = instance::Instance::new(Some(ws_handlers))?;
        let surface = surface::Surface::new(instance.clone())?;

        let physical_devices_raw = unsafe { instance.enumerate_physical_devices() }?;
        let physical_devices = unsafe {
            PhysicalDevice::new(
                instance.clone(),
                Some(surface.clone()),
                physical_devices_raw,
            )
        }?;

        let physical_device = if let Some(physical_device) = physical_devices.iter().find(|pd| {
            pd.properties.device_type == ash::vk::PhysicalDeviceType::DISCRETE_GPU
                || pd.properties.device_type == ash::vk::PhysicalDeviceType::INTEGRATED_GPU
        }) {
            dbg!(physical_device);
            physical_device.clone()
        } else {
            return Err(Box::new(DeviceError::new(
                "no supported GPU's found".to_string(),
            )));
        };

        let (device_handler, extensions) = physical_device
            .create_device(vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true))?;

        let device = Arc::new(Self {
            handle: device_handler,
            instance,
            surface,
            physical_device,
            extensions,
        });

        Ok(device)
    }
}
