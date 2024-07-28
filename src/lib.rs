#[derive(Default, Debug)]
pub struct Mask {
    /// For simplicity we just use `u8` instead of
    /// a proper bitmask.
    /// A value of 1 means allowed and 0 forbidden.
    pub inner: Vec<u8>,
}

impl Mask {
    pub fn new(size: usize) -> Self {
        Mask {
            inner: vec![1; size],
        }
    }
}
