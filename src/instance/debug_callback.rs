use std::error::Error;
use std::ffi::c_void;

use ash::vk::{
    self, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
    DebugUtilsMessengerCallbackDataEXT,
};

use ash::ext::debug_utils;
use ash::{Entry, Instance};

pub(crate) struct DebugUtilsMessanger {
    _debug_utils: debug_utils::Instance,
    _debug_utils_messanger: vk::DebugUtilsMessengerEXT,
}

impl Drop for DebugUtilsMessanger {
    fn drop(&mut self) {
        unsafe {
            self._debug_utils
                .destroy_debug_utils_messenger(self._debug_utils_messanger, None);
        }
    }
}

#[allow(unused)]
unsafe extern "system" fn debug_callback(
    message_severity: DebugUtilsMessageSeverityFlagsEXT,
    message_type: DebugUtilsMessageTypeFlagsEXT,
    callback_data_ptr: *const DebugUtilsMessengerCallbackDataEXT,
    user_data_ptr: *mut c_void,
) -> u32 {
    let callback_data = unsafe { callback_data_ptr.read() };
    match unsafe { callback_data.message_as_c_str() } {
        Some(cstr) => {
            let message = cstr.to_str().unwrap();

            if message_severity.contains(DebugUtilsMessageSeverityFlagsEXT::ERROR) {
                #[allow(clippy::panic)]
                {
                    println!("[FATAL] {message}");
                    let backtrace = std::backtrace::Backtrace::capture();
                    println!("BACKTRACE:\n{}", backtrace);
                    std::process::exit(1)
                };
            } else if message_severity.contains(DebugUtilsMessageSeverityFlagsEXT::INFO) {
                println!("[VALIDATION INFO] {message}");
            } else if message_severity.contains(DebugUtilsMessageSeverityFlagsEXT::VERBOSE) {
                println!("[VALIDATION VERBOSE] {message}");
            } else if message_severity.contains(DebugUtilsMessageSeverityFlagsEXT::WARNING) {
                println!("[VALIDATION WARNING] {message}");
            }
        }
        None => println!("debug callback was called, but invalid callback data was provided"),
    }
    0
}

#[allow(unused)]
pub(crate) fn create_debug_utils_messanger(
    entry: &Entry,
    instance: &Instance,
) -> Result<DebugUtilsMessanger, Box<dyn Error>> {
    let debug_utils = debug_utils::Instance::new(entry, instance);
    let debug_messanger_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            DebugUtilsMessageSeverityFlagsEXT::ERROR
                | DebugUtilsMessageSeverityFlagsEXT::WARNING
                | DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            DebugUtilsMessageTypeFlagsEXT::GENERAL
                | DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback));

    let debug_utils_messanger =
        unsafe { debug_utils.create_debug_utils_messenger(&debug_messanger_create_info, None) }?;

    Ok(DebugUtilsMessanger {
        _debug_utils: debug_utils,
        _debug_utils_messanger: debug_utils_messanger,
    })
}
