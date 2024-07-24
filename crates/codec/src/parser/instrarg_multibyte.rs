use uuasm_nodes::IR;

use crate::{Parse, ParseError};

pub struct InstrArgMultibyteParser<T: IR> {
    data: Vec<T::Instr>,
}

impl<T: IR> Default for InstrArgMultibyteParser<T> {
    fn default() -> Self {
        Self {
            data: Vec::with_capacity(1),
        }
    }
}

impl<T: IR> Parse<T> for InstrArgMultibyteParser<T> {
    type Production = Vec<T::Instr>;

    fn advance(
        &mut self,
        irgen: &mut T,
        window: crate::window::DecodeWindow,
    ) -> crate::ParseResult<T> {
        todo!()
    }

    fn production(self, irgen: &mut T) -> Result<Self::Production, ParseError<<T as IR>::Error>> {
        todo!()
    }
}
