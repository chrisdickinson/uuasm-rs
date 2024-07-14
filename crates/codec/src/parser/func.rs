use uuasm_nodes::IR;

use crate::{Advancement, IRError, Parse};

use super::any::AnyParser;

#[derive(Default)]
pub enum FuncParser<T: IR> {
    #[default]
    Init,

    Locals(Box<[T::Local]>),
    Ready(Box<[T::Local]>, T::Expr),
}

impl<T: IR> Parse<T> for FuncParser<T> {
    type Production = T::Func;

    fn advance(
        &mut self,
        _irgen: &mut T,
        window: crate::window::DecodeWindow,
    ) -> crate::ParseResult<T> {
        match self {
            Self::Init => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::LocalList(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::LocalList(parser) = last_state else {
                        unreachable!();
                    };

                    let locals = parser.production(irgen)?;

                    Ok(AnyParser::Func(Self::Locals(locals)))
                },
            )),

            Self::Locals(_) => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::Expr(Default::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::Expr(parser) = last_state else {
                        unreachable!();
                    };

                    let AnyParser::Func(Self::Locals(locals)) = this_state else {
                        unreachable!();
                    };

                    let expr = parser.production(irgen)?;

                    Ok(AnyParser::Func(Self::Ready(locals, expr)))
                },
            )),

            Self::Ready(_, _) => Ok(Advancement::Ready(window.offset())),
        }
    }

    fn production(
        self,
        irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseError<<T as IR>::Error>> {
        let Self::Ready(locals, expr) = self else {
            unreachable!()
        };

        Ok(irgen.make_func(locals, expr).map_err(IRError)?)
    }
}
