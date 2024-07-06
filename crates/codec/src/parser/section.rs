use uuasm_nodes::IR;

use crate::{window::DecodeWindow, Advancement, IRError, Parse, ParseError, ParseResult};

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

    fn advance(&mut self, _irgen: &mut T, mut window: DecodeWindow) -> ParseResult<T> {
        loop {
            match self {
                SectionParser::ParseType => {
                    *self = SectionParser::ParseLength(window.take()?);
                }

                SectionParser::ParseLength(_xs) => {
                    return Ok(Advancement::YieldTo(
                        window.offset(),
                        AnyParser::LEBU32(LEBParser::new()),
                        |irgen, last_state, this_state| {
                            let AnyParser::LEBU32(leb) = last_state else {
                                unreachable!()
                            };
                            let AnyParser::Section(SectionParser::ParseLength(kind)) = this_state
                            else {
                                unreachable!()
                            };

                            let len = leb.production(irgen)?;
                            Ok(AnyParser::Section(SectionParser::SectionContent(kind, len)))
                        },
                    ));
                }

                SectionParser::SectionContent(kind, length) => {
                    let length = *length as usize;
                    return Ok(match *kind {
                        0x0 => Advancement::YieldTo(
                            window.offset(),
                            AnyParser::Accumulate(Accumulator::new(length)),
                            |irgen, last_state, _| {
                                let AnyParser::Accumulate(acc) = last_state else {
                                    unreachable!();
                                };

                                let bytes = acc.production(irgen)?;
                                Ok(AnyParser::Section(SectionParser::Done(
                                    irgen.make_custom_section(bytes).map_err(IRError)?,
                                )))
                            },
                        ),

                        0x1 => Advancement::YieldTo(
                            window.offset(),
                            AnyParser::TypeSection(Repeated::default()),
                            |irgen, last_state, _| {
                                let AnyParser::TypeSection(ts) = last_state else {
                                    unreachable!();
                                };

                                let items = ts.production(irgen)?;
                                let section = irgen.make_type_section(items).map_err(IRError)?;
                                Ok(AnyParser::Section(SectionParser::Done(section)))
                            },
                        ),

                        0x2 => Advancement::YieldTo(
                            window.offset(),
                            AnyParser::ImportSection(Repeated::default()),
                            |irgen, last_state, _| {
                                let AnyParser::ImportSection(ts) = last_state else {
                                    unreachable!();
                                };

                                let items = ts.production(irgen)?;
                                let section = irgen.make_import_section(items).map_err(IRError)?;
                                Ok(AnyParser::Section(SectionParser::Done(section)))
                            },
                        ),

                        0x3 => Advancement::YieldTo(
                            window.offset(),
                            AnyParser::FunctionSection(Repeated::default()),
                            |irgen, last_state, _| {
                                let AnyParser::FunctionSection(ts) = last_state else {
                                    unreachable!();
                                };

                                let items = ts.production(irgen)?;
                                let section =
                                    irgen.make_function_section(items).map_err(IRError)?;
                                Ok(AnyParser::Section(SectionParser::Done(section)))
                            },
                        ),

                        0x4 => Advancement::YieldTo(
                            window.offset(),
                            AnyParser::TableSection(Repeated::default()),
                            |irgen, last_state, _| {
                                let AnyParser::TableSection(ts) = last_state else {
                                    unreachable!();
                                };

                                let items = ts.production(irgen)?;
                                let section = irgen.make_table_section(items).map_err(IRError)?;
                                Ok(AnyParser::Section(SectionParser::Done(section)))
                            },
                        ),

                        unk => {
                            return Err(ParseError::SectionInvalid {
                                kind: unk,
                                position: window.position(),
                            });
                        }
                    });
                }
                SectionParser::Done(_) => return Ok(Advancement::Ready(window.offset())),
            };
        }
    }

    fn production(self, _irgen: &mut T) -> Result<Self::Production, ParseError<T::Error>> {
        let Self::Done(section_type) = self else {
            unreachable!();
        };

        Ok(section_type)
    }
}
