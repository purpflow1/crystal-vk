use std::error::Error;

use ash::vk;

use crate::render::surface::window::WindowSystemRawHandlers;

pub(super) fn new_in(
    entry: &ash::Entry,
    ws_handlers: Option<WindowSystemRawHandlers>,
) -> Result<ash::Instance, Box<dyn Error>> {
    let mut instance_extensions = vec![
        #[cfg(debug_assertions)]
        vk::EXT_DEBUG_UTILS_NAME.as_ptr(),
        #[cfg(target_vendor = "apple")]
        vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr(),
    ];

    let display_extensions;

    if let Some(ws_handlers) = ws_handlers {
        display_extensions = ash_window::enumerate_required_extensions(ws_handlers.display)?;

        display_extensions
            .iter()
            .for_each(|ext| instance_extensions.push(*ext));
    }

    let flags = if cfg!(target_vendor = "apple") {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };

    pub(super) const API_VERSION_LATEST: u32 = u32::MAX;

    let app_info = vk::ApplicationInfo::default().api_version(API_VERSION_LATEST);
    #[cfg(not(debug_assertions))]
    let create_info = vk::InstanceCreateInfo::default()
        .flags(flags)
        .application_info(&app_info)
        .enabled_extension_names(&instance_extensions);

    #[cfg(debug_assertions)]
    let layers;
    #[cfg(debug_assertions)]
    let layers_pp: Vec<*const i8>;

    #[cfg(debug_assertions)]
    let create_info = {
        use crate::instance::layers;

        layers = layers::get_supported_validation_layers(entry);

        if layers.is_empty() {
            println!(
                "No validation layers found!
                Vulkan SDK should be installed for proper debug.
                Visit https://vulkan.lunarg.com/"
            );

            vk::InstanceCreateInfo::default()
                .flags(flags)
                .application_info(&app_info)
                .enabled_extension_names(&instance_extensions)
        } else {
            layers_pp = layers.iter().map(|x| x.as_ptr() as *const _).collect();

            let mut create_info = vk::InstanceCreateInfo::default()
                .flags(flags)
                .application_info(&app_info)
                .enabled_extension_names(&instance_extensions);

            create_info.pp_enabled_layer_names = layers_pp.as_ptr() as *const _;
            create_info.enabled_layer_count = layers_pp.len() as u32;

            create_info
        }
    };

    let instance = unsafe { entry.create_instance(&create_info, None) }?;
    Ok(instance)
}
