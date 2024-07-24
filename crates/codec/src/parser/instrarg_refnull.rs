use uuasm_nodes::IR;

use crate::{window::DecodeWindow, Advancement, Parse, ParseError, ParseResult};

#[derive(Default)]
pub struct InstrArgRefNullParser {
    data: u8,
}

impl<T: IR> Parse<T> for InstrArgRefNullParser {
    type Production = u8;

    fn advance(&mut self, _irgen: &mut T, mut window: DecodeWindow) -> ParseResult<T> {
        self.data = window.take()?;
        Ok(Advancement::Ready(window.offset()))
    }

    fn production(self, _irgen: &mut T) -> Result<Self::Production, ParseError<<T as IR>::Error>> {
        Ok(self.data)
    }
}
