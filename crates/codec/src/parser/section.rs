use uuasm_nodes::IR;

use crate::{window::DecodeWindow, Advancement, IRError, Parse, ParseErrorKind, ParseResult};

use super::{accumulator::Accumulator, any::AnyParser, leb::LEBParser, repeated::Repeated};

#[derive(Default)]
pub enum SectionParser<T: IR> {
    #[default]
    ParseType,
    ParseLength(u8),
    SectionContent(u8, u32),
    Done(T::Section),
}

impl<T: IR> Parse<T> for SectionParser<T> {
    type Production = T::Section;

    fn advance(&mut self, irgen: &mut T, mut window: &mut DecodeWindow) -> ParseResult<T> {
        loop {
            match self {
                SectionParser::ParseType => {
                    *self = SectionParser::ParseLength(window.take()?);
                }

                SectionParser::ParseLength(_xs) => {
                    return Ok(Advancement::YieldTo(
                        AnyParser::LEBU32(LEBParser::new()),
                        |irgen, last_state, this_state| {
                            let AnyParser::LEBU32(leb) = last_state else {
                                unsafe {
                                    crate::cold();
                                    std::hint::unreachable_unchecked()
                                }
                            };
                            let AnyParser::Section(SectionParser::ParseLength(kind)) = this_state
                            else {
                                unsafe {
                                    crate::cold();
                                    std::hint::unreachable_unchecked()
                                }
                            };

                            let len = leb.production(irgen)?;
                            Ok(AnyParser::Section(SectionParser::SectionContent(kind, len)))
                        },
                    ));
                }

                SectionParser::SectionContent(kind, length) => {
                    let length = *length as usize;
                    irgen.start_section(*kind, length as u32);
                    return Ok(match *kind {
                        0x0 => Advancement::YieldTo(
                            AnyParser::Accumulate(Accumulator::new(length)),
                            |irgen, last_state, _| {
                                let AnyParser::Accumulate(acc) = last_state else {
                                    unsafe {
                                        crate::cold();
                                        std::hint::unreachable_unchecked()
                                    };
                                };

                                let bytes = acc.production(irgen)?;
                                Ok(AnyParser::Section(SectionParser::Done(
                                    irgen.make_custom_section(bytes).map_err(IRError)?,
                                )))
                            },
                        ),

                        0x1 => Advancement::YieldTo(
                            AnyParser::TypeSection(Repeated::default()),
                            |irgen, last_state, _| {
                                let AnyParser::TypeSection(ts) = last_state else {
                                    unsafe {
                                        crate::cold();
                                        std::hint::unreachable_unchecked()
                                    };
                                };

                                let items = ts.production(irgen)?;
                                let section = irgen.make_type_section(items).map_err(IRError)?;
                                Ok(AnyParser::Section(SectionParser::Done(section)))
                            },
                        ),

                        0x2 => Advancement::YieldTo(
                            AnyParser::ImportSection(Repeated::default()),
                            |irgen, last_state, _| {
                                let AnyParser::ImportSection(ts) = last_state else {
                                    unsafe {
                                        crate::cold();
                                        std::hint::unreachable_unchecked()
                                    };
                                };

                                let items = ts.production(irgen)?;
                                let section = irgen.make_import_section(items).map_err(IRError)?;
                                Ok(AnyParser::Section(SectionParser::Done(section)))
                            },
                        ),

                        0x3 => Advancement::YieldTo(
                            AnyParser::FunctionSection(Repeated::default()),
                            |irgen, last_state, _| {
                                let AnyParser::FunctionSection(ts) = last_state else {
                                    unsafe {
                                        crate::cold();
                                        std::hint::unreachable_unchecked()
                                    };
                                };

                                let items = ts.production(irgen)?;
                                let section =
                                    irgen.make_function_section(items).map_err(IRError)?;
                                Ok(AnyParser::Section(SectionParser::Done(section)))
                            },
                        ),

                        0x4 => Advancement::YieldTo(
                            AnyParser::TableSection(Repeated::default()),
                            |irgen, last_state, _| {
                                let AnyParser::TableSection(ts) = last_state else {
                                    unsafe {
                                        crate::cold();
                                        std::hint::unreachable_unchecked()
                                    };
                                };

                                let items = ts.production(irgen)?;
                                let section = irgen.make_table_section(items).map_err(IRError)?;
                                Ok(AnyParser::Section(SectionParser::Done(section)))
                            },
                        ),

                        0x5 => Advancement::YieldTo(
                            AnyParser::MemorySection(Repeated::default()),
                            |irgen, last_state, _| {
                                let AnyParser::MemorySection(ts) = last_state else {
                                    unsafe {
                                        crate::cold();
                                        std::hint::unreachable_unchecked()
                                    };
                                };

                                let items = ts.production(irgen)?;
                                let section = irgen.make_memory_section(items).map_err(IRError)?;
                                Ok(AnyParser::Section(SectionParser::Done(section)))
                            },
                        ),

                        0x6 => Advancement::YieldTo(
                            AnyParser::GlobalSection(Repeated::default()),
                            |irgen, last_state, _| {
                                let AnyParser::GlobalSection(ts) = last_state else {
                                    unsafe {
                                        crate::cold();
                                        std::hint::unreachable_unchecked()
                                    };
                                };

                                let items = ts.production(irgen)?;
                                let section = irgen.make_global_section(items).map_err(IRError)?;
                                Ok(AnyParser::Section(SectionParser::Done(section)))
                            },
                        ),

                        0x7 => Advancement::YieldTo(
                            AnyParser::ExportSection(Repeated::default()),
                            |irgen, last_state, _| {
                                let AnyParser::ExportSection(ts) = last_state else {
                                    unsafe {
                                        crate::cold();
                                        std::hint::unreachable_unchecked()
                                    };
                                };

                                let items = ts.production(irgen)?;
                                let section = irgen.make_export_section(items).map_err(IRError)?;
                                Ok(AnyParser::Section(SectionParser::Done(section)))
                            },
                        ),

                        0x8 => Advancement::YieldTo(
                            AnyParser::LEBU32(Default::default()),
                            |irgen, last_state, _| {
                                let AnyParser::LEBU32(ts) = last_state else {
                                    unsafe {
                                        crate::cold();
                                        std::hint::unreachable_unchecked()
                                    };
                                };

                                let func_idx = ts.production(irgen)?;
                                let func_idx = irgen.make_func_index(func_idx).map_err(IRError)?;

                                let section =
                                    irgen.make_start_section(func_idx).map_err(IRError)?;
                                Ok(AnyParser::Section(SectionParser::Done(section)))
                            },
                        ),

                        0x9 => Advancement::YieldTo(
                            AnyParser::ElementSection(Repeated::default()),
                            |irgen, last_state, _| {
                                let AnyParser::ElementSection(ts) = last_state else {
                                    unsafe {
                                        crate::cold();
                                        std::hint::unreachable_unchecked()
                                    };
                                };

                                let items = ts.production(irgen)?;
                                let section = irgen.make_element_section(items).map_err(IRError)?;
                                Ok(AnyParser::Section(SectionParser::Done(section)))
                            },
                        ),

                        0xa => Advancement::YieldTo(
                            AnyParser::CodeSection(Repeated::default()),
                            |irgen, last_state, _| {
                                let AnyParser::CodeSection(ts) = last_state else {
                                    unsafe {
                                        crate::cold();
                                        std::hint::unreachable_unchecked()
                                    };
                                };

                                let items = ts.production(irgen)?;
                                let section = irgen.make_code_section(items).map_err(IRError)?;
                                Ok(AnyParser::Section(SectionParser::Done(section)))
                            },
                        ),

                        0xb => Advancement::YieldTo(
                            AnyParser::DataSection(Repeated::default()),
                            |irgen, last_state, _| {
                                let AnyParser::DataSection(ts) = last_state else {
                                    unsafe {
                                        crate::cold();
                                        std::hint::unreachable_unchecked()
                                    };
                                };

                                let items = ts.production(irgen)?;
                                let section = irgen.make_data_section(items).map_err(IRError)?;
                                Ok(AnyParser::Section(SectionParser::Done(section)))
                            },
                        ),

                        0xc => Advancement::YieldTo(
                            AnyParser::LEBU32(Default::default()),
                            |irgen, last_state, _| {
                                let AnyParser::LEBU32(ts) = last_state else {
                                    unsafe {
                                        crate::cold();
                                        std::hint::unreachable_unchecked()
                                    };
                                };

                                let count = ts.production(irgen)?;

                                let section =
                                    irgen.make_datacount_section(count).map_err(IRError)?;
                                Ok(AnyParser::Section(SectionParser::Done(section)))
                            },
                        ),

                        unk => {
                            return Err(ParseErrorKind::SectionInvalid {
                                kind: unk,
                                position: window.position(),
                            });
                        }
                    });
                }
                SectionParser::Done(_) => return Ok(Advancement::Ready),
            };
        }
    }

    fn production(self, _irgen: &mut T) -> Result<Self::Production, ParseErrorKind<T::Error>> {
        let Self::Done(section_type) = self else {
            unsafe {
                crate::cold();
                std::hint::unreachable_unchecked()
            };
        };

        Ok(section_type)
    }
}
