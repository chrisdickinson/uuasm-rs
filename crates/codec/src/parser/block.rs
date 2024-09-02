use uuasm_ir::IR;

use crate::{window::DecodeWindow, Advancement, IRError, Parse, ParseErrorKind};

use super::any::AnyParser;

pub enum BlockParser<T: IR> {
    Init(u8),

    BlockType(u8, T::BlockType),

    Ready(T::BlockType, T::Expr),
}

impl<T: IR> Parse<T> for BlockParser<T> {
    type Production = (T::BlockType, T::Expr);

    fn advance(&mut self, irgen: &mut T, _window: &mut DecodeWindow) -> crate::ParseResult<T> {
        Ok(match self {
            Self::Init(_) => Advancement::YieldTo(
                AnyParser::BlockType(Default::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::BlockType(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        }
                    };
                    let block_type = parser.production(irgen)?;

                    let AnyParser::Block(BlockParser::Init(opcode)) = this_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        }
                    };

                    Ok(AnyParser::Block(Self::BlockType(opcode, block_type)))
                },
            ),

            Self::BlockType(opcode, block_type) => {
                // TODO: thread "loop or block" discriminant info through here so we can call the
                // right function
                if *opcode == 0x02 {
                    irgen.start_block(block_type).map_err(IRError)?;
                } else {
                    irgen.start_loop(block_type).map_err(IRError)?;
                }
                Advancement::YieldTo(
                    AnyParser::Expr(Default::default()),
                    |irgen, last_state, this_state| {
                        let AnyParser::Expr(parser) = last_state else {
                            unsafe {
                                crate::cold();
                                std::hint::unreachable_unchecked()
                            }
                        };
                        let AnyParser::Block(Self::BlockType(_, block_type)) = this_state else {
                            unsafe {
                                crate::cold();
                                std::hint::unreachable_unchecked()
                            }
                        };
                        let expr = parser.production(irgen)?;

                        Ok(AnyParser::Block(Self::Ready(block_type, expr)))
                    },
                )
            }

            // TODO: switch to Expr::no_shift() and check that the last byte was 0x0b
            Self::Ready(_, _) => Advancement::Ready,
        })
    }

    fn production(
        self,
        _irgen: &mut T,
    ) -> Result<Self::Production, ParseErrorKind<<T as IR>::Error>> {
        let Self::Ready(block_type, expr) = self else {
            unsafe {
                crate::cold();
                std::hint::unreachable_unchecked()
            }
        };

        Ok((block_type, expr))
    }
}
