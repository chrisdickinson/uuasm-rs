use uuasm_nodes::ImportDesc;

use crate::{window::DecodeWindow, Advancement, Parse, ParseError, ParseResult};

use super::{any::AnyParser, leb::LEBParser};

#[derive(Default)]
pub struct ImportDescParser {
    desc: Option<ImportDesc>,
}

impl Parse for ImportDescParser {
    type Production = ImportDesc;

    fn advance(&mut self, mut window: DecodeWindow) -> ParseResult {
        if self.desc.is_some() {
            return Ok(Advancement::Ready(window.offset()));
        }

        Ok(match window.take()? {
            0x00 => Advancement::YieldTo(
                window.offset(),
                AnyParser::LEBU32(LEBParser::default()),
                |last_state, _| {
                    let AnyParser::LEBU32(parser) = last_state else {
                        unreachable!();
                    };

                    let type_idx = parser.production()?;

                    Ok(AnyParser::ImportDesc(ImportDescParser {
                        desc: Some(ImportDesc::Func(uuasm_nodes::TypeIdx(type_idx))),
                    }))
                },
            ),

            xs => return Err(ParseError::InvalidImportDescriptor(xs)),
        })
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        todo!()
    }
}
