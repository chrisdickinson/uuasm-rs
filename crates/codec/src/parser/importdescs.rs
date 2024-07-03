use uuasm_nodes::ImportDesc;

use crate::{window::DecodeWindow, Parse, ParseError, ParseResult};

#[derive(Default)]
pub struct ImportDescParser {}

impl Parse for ImportDescParser {
    type Production = ImportDesc;

    fn advance(&mut self, window: DecodeWindow) -> ParseResult {
        match window.advance() {}
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        todo!()
    }
}
