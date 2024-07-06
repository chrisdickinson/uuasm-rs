use uuasm_nodes::IR;

use crate::Parse;

#[derive(Default)]
pub enum LimitsParser {
    #[default]
    Init,
    LowerBound,
    Bounded,
    GotLowerBound(u32),
    Ready(u32, Option<u32>),
}

impl<T: IR> Parse<T> for LimitsParser {
    type Production = T::Limits;

    fn advance(
        &mut self,
        _irgen: &mut T,
        _window: crate::window::DecodeWindow,
    ) -> crate::ParseResult<T> {
        todo!()
    }

    fn production(
        self,
        _irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseError<<T as IR>::Error>> {
        todo!()
    }
}
