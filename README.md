# Crystal-VK
**Crystal-VK** is an idiomatic, safe, and high-performance graphics wrapper for the Vulkan API, written in Rust.

This library provides convenient abstractions over Vulkan while maintaining its flexibility and control. It is ideal for developers who want to leverage the full power of Vulkan in Rust without delving into the details of manual resource management.

## ✨ Features

- Safe Abstractions: Utilizes Rust's ownership system for safe management of Vulkan object lifetimes (devices, buffers, images, pipelines).
- Idiomatic Rust API: Offers an ergonomic and expressive interface that follows Rust conventions.
- Automatic Resource Management: Allows you to focus on rendering logic, not on manual allocation and deallocation.
- Support for Various Vulkan Features: Includes compute shaders and next-generation synchronization.
- Cross-Platform: Works on Windows, Linux, and macOS (via MoltenVK).

There are plans to add extensibility, in particular: easy integration of third-party libraries and Vulkan extensions.

---

## 🚀 Quick Start
Add `crystal-vk` to your `Cargo.toml` dependencies
```toml
[dependencies]
crystal-vk = "0.0.4"
```
Here is a minimal example showing how to create a Vulkan device:
```rust
use winit::{dpi::LogicalSize, window::Window};
use crystal_vk::device::Device;

pub fn new(event_loop: &winit::event_loop::ActiveEventLoop) {
    // winit window creation
    let window = {
        event_loop
            .create_window(
                Window::default_attributes()
                    .with_inner_size(LogicalSize::new(300, 300)),
            )
            .unwrap()
    };

    // creates logical device from the first element of the list of available devices
    let (device, queues) = Device::with_present(|devices| devices[0].clone(), &window).unwrap();
}
```

---

## 🤝 Contributing
Contributions to crystal-vk are welcome!

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add some amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

## 📄 License
This project is licensed under either the **MIT** or **GPL 3.0** License (at your option). See the LICENSE-MIT and LICENSE-GPL files for details.
