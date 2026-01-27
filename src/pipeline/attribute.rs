#[derive(Clone, Copy)]
pub struct Attribute {
    pub size: usize,
    pub offset: usize,
}

pub trait AttributeDescriptor {
    fn get_attributes() -> &'static [Attribute];
}
