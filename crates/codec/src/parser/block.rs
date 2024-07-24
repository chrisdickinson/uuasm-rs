use uuasm_nodes::IR;

use crate::Parse;

#[derive(Default)]
pub enum BlockParser<T: IR> {
    #[default]
    Init,

    BlockType(T::BlockType),

    Ready(T::BlockType, T::Expr),
}

impl<T: IR> Parse<T> for BlockParser<T> {
    type Production = (T::BlockType, T::Expr);

    fn advance(
        &mut self,
        irgen: &mut T,
        window: crate::window::DecodeWindow,
    ) -> crate::ParseResult<T> {
        todo!()
    }

    fn production(
        self,
        irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseError<<T as IR>::Error>> {
        todo!()
    }
}
