use std::{error::Error, marker::PhantomData};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};

use crate::errors::DeviceError;

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
            Err(Box::new(DeviceError::new(format!(
                "cannot get window system handlers"
            ))))
        }
    }
}

pub(crate) struct NullWindow {
    _pd: PhantomData<()>,
}

impl HasWindowHandle for NullWindow {
    fn window_handle(
        &self,
    ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        Err(raw_window_handle::HandleError::Unavailable)
    }
}

impl HasDisplayHandle for NullWindow {
    fn display_handle(
        &self,
    ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        Err(raw_window_handle::HandleError::Unavailable)
    }
}
