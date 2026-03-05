//! ## Crystal VK
//! This library provides convenient abstractions over Vulkan while maintaining its
//! flexibility and control. It is ideal for developers who want to leverage the
//! full power of Vulkan in Rust without delving into the details of manual resource management.

pub mod acceleration;
pub mod buffer;
pub mod command;
pub mod deferred;
pub mod device;
pub mod image;
pub mod instance;
pub mod pipeline;
pub mod render;
pub mod sync;

pub use ash::vk;
