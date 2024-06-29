use std::marker::PhantomData;

use crate::{window::DecodeWindow, Advancement, Parse, ParseError, ParseResult};

#[derive(Default)]
pub struct LEBParser<T> {
    repr: u64,
    offs: usize,
    _marker: PhantomData<T>,
}

impl<T: num::Integer + Default> LEBParser<T> {
    pub(crate) fn new() -> Self {
        Self {
            repr: Default::default(),
            offs: 0,
            _marker: PhantomData,
        }
    }
}

impl<T: LEBConstants> Parse for LEBParser<T> {
    type Production = T;

    fn advance(&mut self, mut window: DecodeWindow) -> ParseResult {
        let mut next;
        let mut shift = self.offs * 7;
        while {
            next = window.peek()?;

            self.repr |= ((next & 0x7f) as u64) << shift;
            shift += 7;

            next & 0x80 != 0
        } {
            window.take().unwrap();
        }
        window.take().unwrap();

        Ok(Advancement::Ready(window.offset()))
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        Ok(T::from_u64(self.repr))
    }
}

trait LEBConstants {
    const MAX_BYTES: usize;
    const SIGNED: bool = false;
    fn from_u64(i: u64) -> Self;
}

impl LEBConstants for u32 {
    const MAX_BYTES: usize = 5;
    #[inline]
    fn from_u64(i: u64) -> Self {
        i as u32
    }
}

impl LEBConstants for u64 {
    const MAX_BYTES: usize = 10;
    #[inline]
    fn from_u64(i: u64) -> Self {
        i
    }
}

impl LEBConstants for i32 {
    const MAX_BYTES: usize = 5;
    const SIGNED: bool = true;
    #[inline]
    fn from_u64(i: u64) -> Self {
        i as i32
    }
}

impl LEBConstants for i64 {
    const MAX_BYTES: usize = 10;
    const SIGNED: bool = true;
    #[inline]
    fn from_u64(i: u64) -> Self {
        i as i64
    }
}
