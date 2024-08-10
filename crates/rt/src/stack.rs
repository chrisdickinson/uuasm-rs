#[derive(Default)]
pub(crate) struct Stack {
    storage: Vec<u32>,
}

impl Stack {
    pub(crate) fn new() -> Self {
        Default::default()
    }

    pub(crate) fn take(&mut self) -> u32 {
        let idx = self.storage.len().saturating_sub(1);
        let v = self.storage[idx];
        self.storage.truncate(idx);
        v
    }

    pub(crate) fn take2(&mut self) -> u64 {
        let idx = self.storage.len().saturating_sub(2);
        let v: [u32; 2] = (&self.storage[idx..]).try_into().expect("heck");
        let v = u64::from_ne_bytes(unsafe { std::mem::transmute::<[u32; 2], [u8; 8]>(v) });
        self.storage.truncate(idx);
        v
    }

    pub(crate) fn take4(&mut self) -> u128 {
        let idx = self.storage.len().saturating_sub(4);
        let v: [u32; 4] = (&self.storage[idx..]).try_into().expect("heck");
        let v = u128::from_ne_bytes(unsafe { std::mem::transmute::<[u32; 4], [u8; 16]>(v) });
        self.storage.truncate(idx);
        v
    }

    pub(crate) fn push(&mut self, val: u32) {
        self.storage.push(val);
    }

    pub(crate) fn push2(&mut self, val: u64) {
        self.storage
            .extend(unsafe { std::mem::transmute::<[u8; 8], [u32; 2]>(val.to_ne_bytes()) });
    }

    pub(crate) fn push4(&mut self, val: u128) {
        self.storage
            .extend(unsafe { std::mem::transmute::<[u8; 16], [u32; 4]>(val.to_ne_bytes()) });
    }
}
