use uuasm_ir::IR;

use crate::{window::DecodeWindow, Advancement, IRError, Parse, ParseErrorKind, ParseResult};

use super::any::AnyParser;

#[derive(Default)]
pub enum BlockTypeParser<T: IR> {
    #[default]
    Init,

    Ready(T::BlockType),
}

impl<T: IR> Parse<T> for BlockTypeParser<T> {
    type Production = T::BlockType;

    fn advance(&mut self, irgen: &mut T, window: &mut DecodeWindow) -> ParseResult<T> {
        if matches!(self, Self::Ready(_)) {
            return Ok(Advancement::Ready);
        }

        Ok(match window.peek()? {
            0x40 => {
                window.take().unwrap();
                *self = Self::Ready(irgen.make_block_type_empty().map_err(IRError)?);
                Advancement::Ready
            }

            byt @ 0x6f..=0x7f => {
                window.take().unwrap();
                let val_type = irgen.make_val_type(byt).map_err(IRError)?;
                *self = Self::Ready(irgen.make_block_type_val_type(val_type).map_err(IRError)?);
                Advancement::Ready
            }

            _ => {
                // XXX(chrisdickinson): technically this should parse a U64 value but error if the
                // index is > u32::MAX.
                Advancement::YieldTo(
                    AnyParser::LEBU32(Default::default()),
                    |irgen, last_state, _| {
                        let AnyParser::LEBU32(parser) = last_state else {
                            unsafe {
                                crate::cold();
                                std::hint::unreachable_unchecked()
                            }
                        };
                        let value = parser.production(irgen)?;
                        let type_idx = irgen.make_type_index(value).map_err(IRError)?;
                        let block_type = irgen
                            .make_block_type_type_index(type_idx)
                            .map_err(IRError)?;
                        Ok(AnyParser::BlockType(Self::Ready(block_type)))
                    },
                )
            }
        })
    }

    fn production(
        self,
        _irgen: &mut T,
    ) -> Result<Self::Production, ParseErrorKind<<T as IR>::Error>> {
        let Self::Ready(value) = self else {
            unsafe {
                crate::cold();
                std::hint::unreachable_unchecked()
            }
        };
        Ok(value)
    }
}
