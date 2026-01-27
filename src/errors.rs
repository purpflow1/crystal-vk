use std::fmt::Display;

macro_rules! err {
    ($name:ident) => {
        #[derive(Debug)]
        pub struct $name {
            desc: String,
        }

        impl $name {
            pub fn new(desc: String) -> Self {
                Self { desc }
            }
        }

        impl Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_fmt(format_args!("{self:?}"))
            }
        }

        impl std::error::Error for $name {
            fn description(&self) -> &str {
                self.desc.as_str()
            }
        }
    };
}

#[macro_export]
macro_rules! error {
    ($typ:ident, $desc:expr) => {
        Err(Box::new($typ::new(format!($desc))))
    };
}

err!(DebugError);
err!(DeviceError);
err!(SwapChainError);
err!(MemoryError);
err!(PipelineError);
err!(ImageError);

err!(SyncError);
err!(QueueError);
err!(DescriptorError);
err!(CommandError);

err!(SwapchainOutOfDate);
