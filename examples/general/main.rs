mod new;
mod render;
mod timeline;
mod vulkan_context;
use std::{
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

use vulkan_context::*;
mod watcher;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

#[derive(Default)]
struct ContextWindow {
    render_thread: Option<JoinHandle<()>>,
    window: Option<Arc<Window>>,
    to_stop: Arc<Mutex<bool>>,
    extent: Arc<Mutex<[u32; 2]>>,
}

impl ContextWindow {}

impl ApplicationHandler for ContextWindow {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let (mut ctx, window) = VulkanContext::new(event_loop);
        let window = Arc::new(window);

        let window_thread = window.clone();
        let to_stop_thread = self.to_stop.clone();
        let extent = self.extent.clone();

        let render_thread = std::thread::spawn(move || {
            while !*to_stop_thread.lock().unwrap() {
                let extent = extent.lock().unwrap();
                if extent[0] != 0 {
                    ctx.extent = *extent;
                }
                drop(extent);
                ctx.render(&window_thread).unwrap();
            }
        });

        self.render_thread = Some(render_thread);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                println!("Stopping window context with close request");
                *self.to_stop.lock().unwrap() = true;
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                *self.extent.lock().unwrap() = [size.width, size.height];
            }
            WindowEvent::RedrawRequested => {
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => {
                self.window.as_ref().unwrap().request_redraw();
            }
        }
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        // Wayland surface can be destroyed before vulkan resources removal
        self.render_thread = None;
    }
}

fn main() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut context = ContextWindow::default();
    event_loop
        .run_app(&mut context)
        .expect("cannot run event loop");
}
