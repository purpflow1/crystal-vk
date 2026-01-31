use std::error::Error;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};

#[derive(Clone, Copy)]
pub(crate) struct WindowSystemRawHandlers {
    pub window: RawWindowHandle,
    pub display: RawDisplayHandle,
}

impl WindowSystemRawHandlers {
    pub fn new<T: HasDisplayHandle + HasWindowHandle>(w: &T) -> Result<Self, Box<dyn Error>> {
        if let Ok(window) = w.window_handle()
            && let Ok(display) = w.display_handle()
        {
            Ok(Self {
                window: window.as_raw(),
                display: display.as_raw(),
            })
        } else {
            Err("cannot get window system handlers".into())
        }
    }
}
