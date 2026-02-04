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
    context: Option<Arc<Mutex<VulkanContext>>>,
    window: Option<Arc<Window>>,
    first_run: bool,
}

impl ContextWindow {}

impl ApplicationHandler for ContextWindow {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let (ctx, window) = VulkanContext::new(event_loop);
        let window = Arc::new(window);

        self.context = Some(Arc::new(Mutex::new(ctx)));
        self.render_thread = None;
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let window = self.window.as_ref().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                println!("Stopping window context with close request");
                event_loop.exit();
            }
            WindowEvent::Resized(_size) => {}
            WindowEvent::RedrawRequested => {
                let window_thread = window.clone();
                let ctx = self.context.clone().unwrap();

                if self.first_run {
                    let render_thread = std::thread::spawn(move || {
                        let mut to_stop = false;
                        while !to_stop {
                            if let Ok(mut context) = ctx.lock() {
                                context.render(&window_thread).unwrap();
                                to_stop = context.to_stop;
                                let size = window_thread.inner_size();
                                context.extent = [size.width, size.height]
                            }
                        }
                    });

                    self.render_thread = Some(render_thread);
                    self.first_run = false;
                }

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
        self.context = None;
    }
}

fn main() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut context = ContextWindow {
        first_run: true,
        ..Default::default()
    };
    event_loop
        .run_app(&mut context)
        .expect("cannot run event loop");
}
