use std::{error::Error, sync::Arc};

use ash::khr::deferred_host_operations;
use ash::vk::{self, Handle};

use crate::device::Device;

pub struct DeferredOperation {
    allocator: Arc<DeferredOperationAllocator>,
    operation: vk::DeferredOperationKHR,
}

impl Drop for DeferredOperation {
    fn drop(&mut self) {
        unsafe {
            self.allocator
                .handle
                .destroy_deferred_operation(self.operation, None)
        }
    }
}

impl DeferredOperation {
    pub fn begin(allocator: Arc<DeferredOperationAllocator>) -> Result<Self, Box<dyn Error>> {
        let operation = unsafe { allocator.handle.create_deferred_operation(None)? };

        Ok(Self {
            allocator,
            operation,
        })
    }

    pub fn check_competion(&self) -> Result<bool, Box<dyn Error>> {
        match unsafe {
            self.allocator
                .handle
                .get_deferred_operation_result(self.operation)
        } {
            Ok(()) | Err(vk::Result::SUCCESS) => Ok(true),
            Err(vk::Result::NOT_READY) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    pub fn join(self) -> Result<(), Box<dyn Error>> {
        if !self.operation.is_null() {
            unsafe {
                self.allocator
                    .handle
                    .deferred_operation_join(self.operation)?;
                self.allocator
                    .handle
                    .get_deferred_operation_result(self.operation)?;
            }
        }

        Ok(())
    }
}

pub struct DeferredOperationAllocator {
    handle: deferred_host_operations::Device,
}

impl DeferredOperationAllocator {
    pub fn new(device: Arc<Device>) -> Result<Arc<Self>, Box<dyn Error>> {
        let device = deferred_host_operations::Device::new(
            &device.physical_device.instance.handle,
            &device.handle,
        );
        Ok(Arc::new(Self { handle: device }))
    }
}
