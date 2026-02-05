//! ## 🚀 Quick Start
//! Add `crystal-vk` to your `Cargo.toml` dependencies
//! ```toml
//! [dependencies]
//! crystal-vk = "0.0.4"
//! ```
//! Here is a minimal example showing how to create a Vulkan device:
//! ```rust
//! use winit::{dpi::LogicalSize, window::Window};
//! use crystal_vk::device::Device;
//!
//! pub fn new(event_loop: &winit::event_loop::ActiveEventLoop) {
//!     // winit window creation
//!     let window = {
//!         event_loop
//!             .create_window(
//!                 Window::default_attributes()
//!                     .with_inner_size(LogicalSize::new(300, 300)),
//!             )
//!             .unwrap()
//!     };
//!
//!     // creates logical device from the first element of the list of available devices
//!     let (device, queues) = Device::with_present(|devices| devices[0].clone(), &window).unwrap();
//! }
//! ```

pub mod buffer;
pub mod command;
pub mod device;
pub mod image;
pub mod instance;
pub mod pipeline;
pub mod render;
pub mod sync;
pub mod traits;

pub use ash::vk;
