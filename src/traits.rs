use std::any::Any;

pub trait CommandBufferBinding: Send + Sync {
    fn as_any(&self) -> &dyn Any;
}
pub trait DescriptorSetBinding {}
