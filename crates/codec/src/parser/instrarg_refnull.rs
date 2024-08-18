use uuasm_ir::IR;

use crate::{window::DecodeWindow, Advancement, Parse, ParseErrorKind, ParseResult};

#[derive(Default)]
pub struct InstrArgRefNullParser {
    data: u8,
}

impl<T: IR> Parse<T> for InstrArgRefNullParser {
    type Production = u8;

    fn advance(&mut self, _irgen: &mut T, window: &mut DecodeWindow) -> ParseResult<T> {
        self.data = window.take()?;
        Ok(Advancement::Ready)
    }

    fn production(
        self,
        _irgen: &mut T,
    ) -> Result<Self::Production, ParseErrorKind<<T as IR>::Error>> {
        Ok(self.data)
    }
}
