use uuasm_nodes::IR;

use crate::{window::DecodeWindow, Advancement, Parse, ParseErrorKind};

use super::any::AnyParser;

#[derive(Default)]
pub enum BlockParser<T: IR> {
    #[default]
    Init,

    BlockType(T::BlockType),

    Ready(T::BlockType, T::Expr),
}

impl<T: IR> Parse<T> for BlockParser<T> {
    type Production = (T::BlockType, T::Expr);

    fn advance(&mut self, _irgen: &mut T, mut window: &mut DecodeWindow) -> crate::ParseResult<T> {
        Ok(match self {
            Self::Init => Advancement::YieldTo(
                AnyParser::BlockType(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::BlockType(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        }
                    };
                    let block_type = parser.production(irgen)?;

                    Ok(AnyParser::Block(Self::BlockType(block_type)))
                },
            ),

            Self::BlockType(_) => Advancement::YieldTo(
                AnyParser::Expr(Default::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::Expr(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        }
                    };
                    let AnyParser::Block(Self::BlockType(block_type)) = this_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        }
                    };
                    let expr = parser.production(irgen)?;

                    Ok(AnyParser::Block(Self::Ready(block_type, expr)))
                },
            ),

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
