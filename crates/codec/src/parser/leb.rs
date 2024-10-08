use std::marker::PhantomData;

use uuasm_ir::IR;

use crate::{window::DecodeWindow, Advancement, Parse, ParseErrorKind, ParseResult};

#[derive(Default)]
pub struct LEBParser<T> {
    repr: u64,
    offs: u8,
    last: u8,
    _marker: PhantomData<T>,
}

impl<T: num::Integer + Default> LEBParser<T> {
    pub(crate) fn new() -> Self {
        Self {
            repr: Default::default(),
            offs: 0,
            last: 0,
            _marker: PhantomData,
        }
    }
}

impl<T: IR, C: LEBConstants> Parse<T> for LEBParser<C> {
    type Production = C;

    fn advance(&mut self, _irgen: &mut T, window: &mut DecodeWindow) -> ParseResult<T> {
        let mut next = 0;
        let mut offs = self.offs;
        let mut repr = self.repr;
        let mut shift = offs * 7;
        while {
            next = match window.peek() {
                Ok(xs) => xs,
                Err(e) => {
                    self.last = next;
                    self.repr = repr;
                    self.offs = offs;
                    return Err(e.into());
                }
            };

            if shift > 63 {
                return Err(ParseErrorKind::LEBTooLong);
            }
            repr |= ((next & 0x7f) as u64) << shift;
            shift += 7;

            next & 0x80 != 0
        } {
            let _ = window.take();
            offs += 1;
        }
        let _ = window.take();
        self.last = next;
        self.repr = repr;
        self.offs = offs;

        Ok(Advancement::Ready)
    }

    fn production(self, _irgen: &mut T) -> Result<Self::Production, ParseErrorKind<T::Error>> {
        C::from_u64(self.repr, self.offs, self.last).map_err(|e| match e {
            LEBError::Overflow => ParseErrorKind::LEBTooBig,
            LEBError::Overlong => ParseErrorKind::LEBTooLong,
        })
    }
}

pub enum LEBError {
    Overflow,
    Overlong,
}

trait LEBConstants {
    fn from_u64(i: u64, offset: u8, last: u8) -> Result<Self, LEBError>
    where
        Self: Sized;
}

impl LEBConstants for u32 {
    #[inline]
    fn from_u64(i: u64, offset: u8, _last: u8) -> Result<Self, LEBError> {
        if offset > 4 {
            return Err(LEBError::Overlong);
        }
        if i as u32 as u64 != i {
            return Err(LEBError::Overflow);
        }
        Ok(i as u32)
    }
}

impl LEBConstants for u64 {
    #[inline]
    fn from_u64(i: u64, _offset: u8, _last: u8) -> Result<Self, LEBError> {
        Ok(i)
    }
}

impl LEBConstants for i32 {
    #[inline]
    fn from_u64(i: u64, offset: u8, last: u8) -> Result<Self, LEBError> {
        if offset > 4 {
            return Err(LEBError::Overlong);
        }
        let mut result = i as i64;
        let shift = offset as usize * 7;
        let sign_requested = last & 0x40 != 0;
        if shift < 57 && sign_requested {
            result |= !0 << (shift + 7);
        }

        if result as i32 as i64 != result {
            return Err(LEBError::Overflow);
        }
        Ok(result as i32)
    }
}

impl LEBConstants for i64 {
    #[inline]
    fn from_u64(i: u64, offset: u8, last: u8) -> Result<Self, LEBError> {
        let mut result = i as i64;
        let shift = offset as usize * 7;
        let sign_requested = last & 0x40 != 0;
        if sign_requested {
            if shift == 63 {
                // we already stored data in the 64th bit
                if last & 0x3f != 0x3f {
                    return Err(LEBError::Overflow);
                }
            } else {
                result |= !0 << (shift + 7);
            }
        } else {
            if shift == 63 {
                // we already stored data in the 64th bit
                if last > 0 {
                    return Err(LEBError::Overflow);
                }
            }
        }
        Ok(result)
    }
}
