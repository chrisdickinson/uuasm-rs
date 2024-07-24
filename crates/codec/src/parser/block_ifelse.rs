use uuasm_nodes::IR;

use crate::{window::DecodeWindow, Advancement, Parse, ParseError, ParseResult};

use super::any::AnyParser;

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

    fn advance(&mut self, _irgen: &mut T, mut window: DecodeWindow) -> ParseResult<T> {
        Ok(match self {
            Self::Init => Advancement::YieldTo(
                window.offset(),
                AnyParser::BlockType(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::BlockType(parser) = last_state else {
                        unreachable!();
                    };

                    let production = parser.production(irgen)?;
                    Ok(AnyParser::IfElseBlock(Self::BlockType(production)))
                },
            ),

            Self::BlockType(_) => Advancement::YieldTo(
                window.offset(),
                AnyParser::Expr(Default::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::Expr(parser) = last_state else {
                        unreachable!();
                    };
                    let AnyParser::IfElseBlock(Self::BlockType(block_type)) = this_state else {
                        unreachable!();
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
                        window.offset(),
                        AnyParser::Expr(Default::default()),
                        |irgen, last_state, this_state| {
                            let AnyParser::Expr(parser) = last_state else {
                                unreachable!();
                            };
                            let AnyParser::IfElseBlock(Self::Consequent(block_type, consequent)) =
                                this_state
                            else {
                                unreachable!();
                            };

                            let production = parser.production(irgen)?;
                            Ok(AnyParser::IfElseBlock(Self::Alternate(
                                block_type, consequent, production,
                            )))
                        },
                    )
                } else if next == 0x0b {
                    Advancement::Ready(window.offset())
                } else {
                    return Err(ParseError::InvalidState(
                        "Got a non-block-terminating value",
                    ));
                }
            }

            Self::Alternate(_, _, _) => Advancement::Ready(window.offset()),
        })
    }

    fn production(self, _irgen: &mut T) -> Result<Self::Production, ParseError<<T as IR>::Error>> {
        Ok(match self {
            IfElseBlockParser::Init | IfElseBlockParser::BlockType(_) => unreachable!(),
            IfElseBlockParser::Consequent(block_type, consequent) => (block_type, consequent, None),
            IfElseBlockParser::Alternate(block_type, consequent, alternate) => {
                (block_type, consequent, Some(alternate))
            }
        })
    }
}
