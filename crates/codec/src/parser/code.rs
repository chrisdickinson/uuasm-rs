use uuasm_ir::IR;

use crate::{window::DecodeWindow, Advancement, IRError, Parse};

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

    fn advance(&mut self, _irgen: &mut T, _window: &mut DecodeWindow) -> crate::ParseResult<T> {
        match self {
            Self::Init => Ok(Advancement::YieldTo(
                AnyParser::LEBU32(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::LEBU32(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let size = parser.production(irgen)?;

                    Ok(AnyParser::Code(Self::GotSize(size)))
                },
            )),

            Self::GotSize(expected) => Ok(Advancement::YieldToBounded(
                *expected,
                AnyParser::Func(FuncParser::default()),
                |irgen, last_state, _| {
                    let AnyParser::Func(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let body = parser.production(irgen)?;

                    Ok(AnyParser::Code(Self::Ready(body)))
                },
            )),

            Self::Ready(_) => Ok(Advancement::Ready),
        }
    }

    fn production(
        self,
        irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseErrorKind<<T as IR>::Error>> {
        let Self::Ready(production) = self else {
            unsafe {
                crate::cold();
                std::hint::unreachable_unchecked()
            }
        };

        Ok(irgen.make_code(production).map_err(IRError)?)
    }
}
