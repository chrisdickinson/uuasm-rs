use uuasm_nodes::IR;

use crate::{window::DecodeWindow, Advancement, IRError, Parse, ParseErrorKind, ParseResult};

use super::any::AnyParser;

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

    fn advance(&mut self, _irgen: &mut T, mut window: &mut DecodeWindow) -> ParseResult<T> {
        match self {
            Self::Init => {
                let kind = window.take()?;
                *self = match kind {
                    0x0 => Self::ImportFunc,
                    0x1 => Self::ImportTable,
                    0x2 => Self::ImportMemory,
                    0x3 => Self::ImportGlobal,
                    unk => return Err(ParseErrorKind::BadImportDesc(unk)),
                };
                self.advance(_irgen, window)
            }

            Self::ImportFunc => Ok(Advancement::YieldTo(
                AnyParser::TypeIdx(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::TypeIdx(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let type_idx = parser.production(irgen)?;
                    let func_import_desc =
                        irgen.make_import_desc_func(type_idx).map_err(IRError)?;

                    Ok(AnyParser::ImportDesc(Self::Ready(func_import_desc)))
                },
            )),

            Self::ImportTable => Ok(Advancement::YieldTo(
                AnyParser::TableType(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::TableType(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let table_type = parser.production(irgen)?;
                    let table_import_desc =
                        irgen.make_import_desc_table(table_type).map_err(IRError)?;

                    Ok(AnyParser::ImportDesc(Self::Ready(table_import_desc)))
                },
            )),

            Self::ImportGlobal => Ok(Advancement::YieldTo(
                AnyParser::GlobalType(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::GlobalType(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let global_type = parser.production(irgen)?;
                    let global_import_desc = irgen
                        .make_import_desc_global(global_type)
                        .map_err(IRError)?;

                    Ok(AnyParser::ImportDesc(Self::Ready(global_import_desc)))
                },
            )),

            Self::ImportMemory => Ok(Advancement::YieldTo(
                AnyParser::Limits(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::Limits(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let limits = parser.production(irgen)?;
                    let memory_import_desc =
                        irgen.make_import_desc_memtype(limits).map_err(IRError)?;

                    Ok(AnyParser::ImportDesc(Self::Ready(memory_import_desc)))
                },
            )),
            Self::Ready(_) => Ok(Advancement::Ready),
        }
    }

    fn production(
        self,
        _irgen: &mut T,
    ) -> Result<Self::Production, ParseErrorKind<<T as IR>::Error>> {
        let Self::Ready(desc) = self else {
            unsafe {
                crate::cold();
                std::hint::unreachable_unchecked()
            }
        };

        Ok(desc)
    }
}
