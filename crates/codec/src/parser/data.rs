use uuasm_nodes::IR;

use crate::{Advancement, Parse, ParseError};

use super::any::AnyParser;

#[derive(Default)]
pub enum DataParser<T: IR> {
    #[default]
    Init,
    Type(u8),
    ActiveUnindexed(T::Expr),

    Ready(T::Data),
}

impl<T: IR> Parse<T> for DataParser<T> {
    type Production = T::Data;

    fn advance(
        &mut self,
        _irgen: &mut T,
        mut window: crate::window::DecodeWindow,
    ) -> crate::ParseResult<T> {
        match self {
            Self::Init => match window.take()? {
                0x00 => Ok(Advancement::YieldTo(
                    window.offset(),
                    AnyParser::Expr(Default::default()),
                    // this parses a constexpr, then a bytevec, becoming a data::active(vec, memidx(0),
                    // expr)
                    |_irgen, _last_state, _this_state| todo!(),
                )),

                // we need a "ByteVec" -- Repeated<> but for just a chunk o'
                // bytes. NameParser could use it too!
                // this parses a bytevec and becomes a data::passive
                0x01 => todo!(),

                // this parses a memory index, const expr, and bytevec; producing
                // an active data segment
                0x02 => todo!(),

                unk => Err(ParseError::BadDataType(unk)),
            },

            Self::Type(_) => todo!(),
            Self::ActiveUnindexed(_) => todo!(),
            Self::Ready(_) => todo!(),
        }
    }

    fn production(
        self,
        _irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseError<<T as IR>::Error>> {
        let Self::Ready(production) = self else {
            unreachable!()
        };

        Ok(production)
    }
}
