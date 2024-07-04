use uuasm_nodes::IR;

use crate::{window::DecodeWindow, Advancement, Parse, ParseError, ParseResult};

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
                            |_irgen, last_state, _| {
                                let AnyParser::Accumulate(_acc) = last_state else {
                                    unreachable!();
                                };

                                todo!("get custom section from IR")
                                //Ok(AnyParser::Section(SectionParser::Done(
                                //    Section::Custom(acc.production(irgen)?),
                                //)))
                            },
                        ),

                        0x1 => Advancement::YieldTo(
                            window.offset(),
                            AnyParser::TypeSection(Repeated::default()),
                            |_irgen, last_state, _| {
                                let AnyParser::TypeSection(_ts) = last_state else {
                                    unreachable!();
                                };

                                todo!("get type section from IR")
                                // Ok(AnyParser::Section(SectionParser::Done(Section::Type(
                                //     ts.production()?,
                                // ))))
                            },
                        ),

                        // 0x1 => Section::Type(Vec::<Type>::from_wasm_bytes(section)?.1),
                        unk => {
                            return Err(ParseError::SectionInvalid {
                                kind: unk,
                                position: window.position(),
                            })
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
