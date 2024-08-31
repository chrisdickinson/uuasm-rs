use uuasm_ir::IR;

use crate::{window::DecodeWindow, Advancement, IRError, Parse, ParseResult};

use super::{any::AnyParser, expr::ExprParser};

#[derive(Default)]
pub enum GlobalParser<T: IR> {
    #[default]
    Init,

    GlobalType(T::GlobalType),
    Ready(T::GlobalType, T::Expr),
}

impl<T: IR> Parse<T> for GlobalParser<T> {
    type Production = T::Global;

    fn advance(&mut self, _irgen: &mut T, _window: &mut DecodeWindow) -> ParseResult<T> {
        match self {
            GlobalParser::Init => Ok(Advancement::YieldTo(
                AnyParser::GlobalType(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::GlobalType(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let production = parser.production(irgen)?;
                    irgen.start_global(&production).map_err(IRError)?;
                    Ok(AnyParser::Global(Self::GlobalType(production)))
                },
            )),
            GlobalParser::GlobalType(_) => Ok(Advancement::YieldTo(
                AnyParser::Expr(ExprParser::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::Expr(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let AnyParser::Global(Self::GlobalType(global_type)) = this_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    Ok(AnyParser::Global(Self::Ready(
                        global_type,
                        parser.production(irgen)?,
                    )))
                },
            )),
            GlobalParser::Ready(_, _) => Ok(Advancement::Ready),
        }
    }

    fn production(
        self,
        irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseErrorKind<<T as IR>::Error>> {
        let Self::Ready(global_type, expr) = self else {
            unsafe {
                crate::cold();
                std::hint::unreachable_unchecked()
            }
        };

        Ok(irgen.make_global(global_type, expr).map_err(IRError)?)
    }
}
