use uuasm_nodes::IR;

use crate::{Advancement, IRError, Parse};

use super::{any::AnyParser, func::FuncParser};

#[derive(Default)]
pub enum CodeParser<T: IR> {
    #[default]
    Init,
    GotSize(u32),
    Ready(T::Func),
}

impl<T: IR> Parse<T> for CodeParser<T> {
    type Production = T::Code;

    fn advance(
        &mut self,
        _irgen: &mut T,
        window: crate::window::DecodeWindow,
    ) -> crate::ParseResult<T> {
        match self {
            Self::Init => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::LEBU32(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::LEBU32(parser) = last_state else {
                        unreachable!();
                    };

                    let size = parser.production(irgen)?;

                    Ok(AnyParser::Code(Self::GotSize(size)))
                },
            )),

            Self::GotSize(expected) => Ok(Advancement::YieldToBounded(
                window.offset(),
                *expected,
                AnyParser::Func(FuncParser::default()),
                |irgen, last_state, _| {
                    let AnyParser::Func(parser) = last_state else {
                        unreachable!();
                    };

                    let body = parser.production(irgen)?;

                    Ok(AnyParser::Code(Self::Ready(body)))
                },
            )),

            Self::Ready(_) => Ok(Advancement::Ready(window.offset())),
        }
    }

    fn production(
        self,
        irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseError<<T as IR>::Error>> {
        let Self::Ready(production) = self else {
            unreachable!()
        };

        Ok(irgen.make_code(production).map_err(IRError)?)
    }
}
