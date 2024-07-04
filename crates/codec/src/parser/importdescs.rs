use uuasm_nodes::IR;

use crate::{window::DecodeWindow, Advancement, Parse, ParseError, ParseResult};

use super::{any::AnyParser, leb::LEBParser};

#[derive(Default)]
pub struct ImportDescParser {
    desc: Option<u32>,
}

impl<T: IR> Parse<T> for ImportDescParser {
    type Production = <T as IR>::ImportDesc;

    fn advance(&mut self, _irgen: &mut T, mut window: DecodeWindow) -> ParseResult<T> {
        if self.desc.is_some() {
            return Ok(Advancement::Ready(window.offset()));
        }

        Ok(match window.take()? {
            0x00 => Advancement::YieldTo(
                window.offset(),
                AnyParser::LEBU32(LEBParser::default()),
                |irgen, last_state, _| {
                    let AnyParser::LEBU32(parser) = last_state else {
                        unreachable!();
                    };

                    let type_idx = parser.production(irgen)?;

                    Ok(AnyParser::ImportDesc(ImportDescParser {
                        desc: Some(type_idx),
                    }))
                },
            ),

            xs => return Err(ParseError::InvalidImportDescriptor(xs)),
        })
    }

    fn production(self, _irgen: &mut T) -> Result<Self::Production, ParseError<T::Error>> {
        todo!()
    }
}

