use uuasm_nodes::IR;

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

    fn advance(&mut self, _irgen: &mut T, mut window: DecodeWindow) -> ParseResult<T> {
        match self {
            GlobalParser::Init => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::GlobalType(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::GlobalType(parser) = last_state else {
                        unreachable!();
                    };

                    let production = parser.production(irgen)?;
                    Ok(AnyParser::Global(Self::GlobalType(production)))
                },
            )),
            GlobalParser::GlobalType(_) => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::Expr(ExprParser::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::Expr(parser) = last_state else {
                        unreachable!();
                    };
                    let AnyParser::Global(Self::GlobalType(global_type)) = this_state else {
                        unreachable!();
                    };

                    Ok(AnyParser::Global(Self::Ready(
                        global_type,
                        parser.production(irgen)?,
                    )))
                },
            )),
            GlobalParser::Ready(_, _) => {
                window.take()?;
                Ok(Advancement::Ready(window.offset()))
            }
        }
    }

    fn production(
        self,
        irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseError<<T as IR>::Error>> {
        let Self::Ready(global_type, expr) = self else {
            unreachable!()
        };

        Ok(irgen.make_global(global_type, expr).map_err(IRError)?)
    }
}
