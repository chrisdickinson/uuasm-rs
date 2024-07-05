use uuasm_nodes::IR;

use crate::{window::DecodeWindow, Advancement, IRError, Parse, ParseError, ParseResult};

use super::{any::AnyParser, leb::LEBParser};

#[derive(Default)]
pub enum ImportDescParser<T: IR> {
    #[default]
    Init,
    ImportFunc,
    ImportTable,
    ImportGlobal,
    ImportMemory,
    Ready(T::ImportDesc),
}

impl<T: IR> Parse<T> for ImportDescParser<T> {
    type Production = <T as IR>::ImportDesc;

    fn advance(&mut self, irgen: &mut T, mut window: DecodeWindow) -> ParseResult<T> {
        match self {
            Self::Init => {
                let kind = window.take()?;
                *self = match kind {
                    0x0 => Self::ImportFunc,
                    0x1 => Self::ImportTable,
                    0x2 => Self::ImportMemory,
                    0x3 => Self::ImportGlobal,
                    unk => return Err(ParseError::BadImportDesc(unk)),
                };
                return self.advance(irgen, window);
            }

            Self::ImportFunc => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::LEBU32(LEBParser::default()),
                |irgen, last_state, _| {
                    let AnyParser::LEBU32(parser) = last_state else {
                        unreachable!();
                    };

                    let type_idx = parser.production(irgen)?;
                    let type_idx = irgen.make_type_index(type_idx).map_err(IRError)?;
                    let func_import_desc =
                        irgen.make_import_desc_func(type_idx).map_err(IRError)?;

                    Ok(AnyParser::ImportDesc(Self::Ready(func_import_desc)))
                },
            )),

            Self::ImportTable => todo!(),
            Self::ImportGlobal => todo!(),
            Self::ImportMemory => todo!(),
            Self::Ready(_) => todo!(),
        }
    }

    fn production(self, irgen: &mut T) -> Result<Self::Production, ParseError<<T as IR>::Error>> {
        todo!()
    }
}

/*
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
*/

