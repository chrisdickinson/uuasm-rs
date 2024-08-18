use uuasm_ir::IR;

use crate::{window::DecodeWindow, Advancement, Parse, ParseErrorKind, ParseResult};

pub struct Accumulator(usize, Box<[u8]>);

impl Accumulator {
    pub(crate) fn new(expected: usize) -> Self {
        Accumulator(0, vec![0; expected].into())
    }
}

impl<T: IR> Parse<T> for Accumulator {
    type Production = Box<[u8]>;

    fn advance(&mut self, _irgen: &mut T, window: &mut DecodeWindow) -> ParseResult<T> {
        let into = &mut self.1[self.0..];
        if !into.is_empty() {
            self.0 += window.take_n(into)?;
        }
        if self.0 == self.1.len() {
            Ok(Advancement::Ready)
        } else {
            Err(ParseErrorKind::Advancement(
                crate::window::AdvancementError::Incomplete(self.1.len() - self.0),
            ))
        }
    }

    fn production(self, _irgen: &mut T) -> Result<Self::Production, ParseErrorKind<T::Error>> {
        Ok(self.1)
    }
}
