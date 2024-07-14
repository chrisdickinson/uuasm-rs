use uuasm_nodes::IR;

use crate::{window::DecodeWindow, Advancement, IRError, Parse, ParseError, ParseResult};

use super::any::AnyParser;

#[derive(Default)]
pub enum ExportDescParser<T: IR> {
    #[default]
    Init,
    ExportFunc,
    ExportTable,
    ExportGlobal,
    ExportMemory,

    Ready(T::ExportDesc),
}

impl<T: IR> Parse<T> for ExportDescParser<T> {
    type Production = T::ExportDesc;

    fn advance(&mut self, _irgen: &mut T, mut window: DecodeWindow) -> ParseResult<T> {
        match self {
            Self::Init => {
                let kind = window.take()?;
                *self = match kind {
                    0x0 => Self::ExportFunc,
                    0x1 => Self::ExportTable,
                    0x2 => Self::ExportMemory,
                    0x3 => Self::ExportGlobal,
                    unk => return Err(ParseError::BadExportDesc(unk)),
                };
                self.advance(_irgen, window)
            }

            Self::ExportFunc => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::FuncIdx(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::FuncIdx(parser) = last_state else {
                        unreachable!();
                    };

                    let func_idx = parser.production(irgen)?;
                    let func_export_desc =
                        irgen.make_export_desc_func(func_idx).map_err(IRError)?;

                    Ok(AnyParser::ExportDesc(Self::Ready(func_export_desc)))
                },
            )),

            Self::ExportTable => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::TableIdx(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::TableIdx(parser) = last_state else {
                        unreachable!();
                    };

                    let table_type = parser.production(irgen)?;
                    let table_export_desc =
                        irgen.make_export_desc_table(table_type).map_err(IRError)?;

                    Ok(AnyParser::ExportDesc(Self::Ready(table_export_desc)))
                },
            )),

            Self::ExportGlobal => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::GlobalIdx(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::GlobalIdx(parser) = last_state else {
                        unreachable!();
                    };

                    let global_idx = parser.production(irgen)?;
                    let global_export_desc =
                        irgen.make_export_desc_global(global_idx).map_err(IRError)?;

                    Ok(AnyParser::ExportDesc(Self::Ready(global_export_desc)))
                },
            )),

            Self::ExportMemory => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::MemIdx(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::MemIdx(parser) = last_state else {
                        unreachable!();
                    };

                    let mem_idx = parser.production(irgen)?;
                    let memory_export_desc =
                        irgen.make_export_desc_memtype(mem_idx).map_err(IRError)?;

                    Ok(AnyParser::ExportDesc(Self::Ready(memory_export_desc)))
                },
            )),
            Self::Ready(_) => Ok(Advancement::Ready(window.offset())),
        }
    }

    fn production(self, _irgen: &mut T) -> Result<Self::Production, ParseError<<T as IR>::Error>> {
        let Self::Ready(production) = self else {
            unreachable!();
        };
        Ok(production)
    }
}
