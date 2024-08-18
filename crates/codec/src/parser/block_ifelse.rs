use uuasm_ir::IR;

use crate::{window::DecodeWindow, Advancement, Parse, ParseErrorKind, ParseResult};

use super::{any::AnyParser, expr::ExprParser};

#[derive(Default)]
pub enum IfElseBlockParser<T: IR> {
    #[default]
    Init,

    BlockType(T::BlockType),

    Consequent(T::BlockType, T::Expr),

    Alternate(T::BlockType, T::Expr, T::Expr),
}

impl<T: IR> Parse<T> for IfElseBlockParser<T> {
    type Production = (T::BlockType, T::Expr, Option<T::Expr>);

    fn advance(&mut self, _irgen: &mut T, window: &mut DecodeWindow) -> ParseResult<T> {
        Ok(match self {
            Self::Init => Advancement::YieldTo(
                AnyParser::BlockType(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::BlockType(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let production = parser.production(irgen)?;
                    Ok(AnyParser::IfElseBlock(Self::BlockType(production)))
                },
            ),

            Self::BlockType(_) => Advancement::YieldTo(
                AnyParser::Expr(ExprParser::no_shift()),
                |irgen, last_state, this_state| {
                    let AnyParser::Expr(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let AnyParser::IfElseBlock(Self::BlockType(block_type)) = this_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let production = parser.production(irgen)?;
                    Ok(AnyParser::IfElseBlock(Self::Consequent(
                        block_type, production,
                    )))
                },
            ),

            Self::Consequent(_, _) => {
                let next = window.take()?;
                if next == 0x05 {
                    Advancement::YieldTo(
                        AnyParser::Expr(Default::default()),
                        |irgen, last_state, this_state| {
                            let AnyParser::Expr(parser) = last_state else {
                                unsafe {
                                    crate::cold();
                                    std::hint::unreachable_unchecked()
                                };
                            };
                            let AnyParser::IfElseBlock(Self::Consequent(block_type, consequent)) =
                                this_state
                            else {
                                unsafe {
                                    crate::cold();
                                    std::hint::unreachable_unchecked()
                                };
                            };

                            let production = parser.production(irgen)?;
                            Ok(AnyParser::IfElseBlock(Self::Alternate(
                                block_type, consequent, production,
                            )))
                        },
                    )
                } else if next == 0x0b {
                    Advancement::Ready
                } else {
                    return Err(ParseErrorKind::InvalidState(
                        "Got a non-block-terminating value",
                    ));
                }
            }

            Self::Alternate(_, _, _) => Advancement::Ready,
        })
    }

    fn production(
        self,
        _irgen: &mut T,
    ) -> Result<Self::Production, ParseErrorKind<<T as IR>::Error>> {
        Ok(match self {
            IfElseBlockParser::Init | IfElseBlockParser::BlockType(_) => unsafe {
                crate::cold();
                std::hint::unreachable_unchecked()
            },
            IfElseBlockParser::Consequent(block_type, consequent) => (block_type, consequent, None),
            IfElseBlockParser::Alternate(block_type, consequent, alternate) => {
                (block_type, consequent, Some(alternate))
            }
        })
    }
}
