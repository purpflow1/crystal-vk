mod new;
mod render;
mod vulkan_context;
use vulkan_context::*;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

#[derive(Default)]
struct ContextWindow {
    data: Option<VulkanContext>,
    window: Option<Window>,
}

impl ContextWindow {}

impl ApplicationHandler for ContextWindow {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let (ctx, window) = VulkanContext::new(event_loop);
        self.data = Some(ctx);
        self.window = Some(window);
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {}

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let data = self.data.as_mut().unwrap();
        let window = self.window.as_ref().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                println!("Stopping window context with close request");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => data.extent = [size.width, size.height],
            WindowEvent::RedrawRequested => {
                data.render(window);
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => {
                self.window.as_ref().unwrap().request_redraw();
            }
        }
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        // Wayland surface can be destroyed before vulkan resources removal
        self.data = None;
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut context = ContextWindow::default();
    event_loop
        .run_app(&mut context)
        .expect("cannot run event loop");
}
