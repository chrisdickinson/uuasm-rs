use uuasm_nodes::IR;

use crate::{window::DecodeWindow, Advancement, Parse, ParseError};

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

    fn advance(&mut self, _irgen: &mut T, mut window: DecodeWindow) -> crate::ParseResult<T> {
        Ok(match self {
            Self::Init => Advancement::YieldTo(
                window.offset(),
                AnyParser::BlockType(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::BlockType(parser) = last_state else {
                         unsafe { crate::cold(); std::hint::unreachable_unchecked() }
                    };
                    let block_type = parser.production(irgen)?;

                    Ok(AnyParser::Block(Self::BlockType(block_type)))
                },
            ),

            Self::BlockType(_) => Advancement::YieldTo(
                window.offset(),
                AnyParser::Expr(Default::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::Expr(parser) = last_state else {
                         unsafe { crate::cold(); std::hint::unreachable_unchecked() }
                    };
                    let AnyParser::Block(Self::BlockType(block_type)) = this_state else {
                         unsafe { crate::cold(); std::hint::unreachable_unchecked() }
                    };
                    let expr = parser.production(irgen)?;

                    Ok(AnyParser::Block(Self::Ready(block_type, expr)))
                },
            ),

            Self::Ready(_, _) => {
                window.take()?;
                Advancement::Ready(window.offset())
            }
        })
    }

    fn production(self, _irgen: &mut T) -> Result<Self::Production, ParseError<<T as IR>::Error>> {
        let Self::Ready(block_type, expr) = self else {
             unsafe { crate::cold(); std::hint::unreachable_unchecked() }
        };

        Ok((block_type, expr))
    }
}
