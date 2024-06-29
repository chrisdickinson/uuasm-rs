use crate::{window::DecodeWindow, Advancement, Parse, ParseError, ParseResult};

pub struct Accumulator(usize, Box<[u8]>);

impl Accumulator {
    pub(crate) fn new(expected: usize) -> Self {
        Accumulator(0, vec![0; expected].into())
    }
}

impl Parse for Accumulator {
    type Production = Box<[u8]>;

    fn advance(&mut self, mut window: DecodeWindow) -> ParseResult {
        let into = &mut self.1[self.0..];
        if !into.is_empty() {
            self.0 += window.take_n(into)?;
        }
        if self.0 == self.1.len() {
            Ok(Advancement::Ready(window.offset()))
        } else {
            Err(ParseError::Incomplete(self.1.len() - self.0))
        }
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        Ok(self.1)
    }
}
