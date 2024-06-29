use uuasm_nodes::SectionType;

use crate::{window::DecodeWindow, Advancement, Parse, ParseError, ParseResult};

use super::{
    accumulator::Accumulator, any::AnyParser, leb::LEBParser, take::Take, types::TypeParser,
};

#[derive(Default)]
pub enum SectionParser {
    #[default]
    ParseType,
    ParseLength(u8),
    SectionContent(u8, u32),
    Done(SectionType),
}

impl Parse for SectionParser {
    type Production = SectionType;

    fn advance(&mut self, mut window: DecodeWindow) -> ParseResult {
        loop {
            match self {
                SectionParser::ParseType => {
                    *self = SectionParser::ParseLength(window.take()?);
                }

                SectionParser::ParseLength(_xs) => {
                    return Ok(Advancement::YieldTo(
                        window.offset(),
                        AnyParser::LEBU32(LEBParser::new()),
                        |last_state, this_state| {
                            let AnyParser::LEBU32(leb) = last_state else {
                                unreachable!()
                            };
                            let AnyParser::Section(SectionParser::ParseLength(kind)) = this_state
                            else {
                                unreachable!()
                            };

                            let len = leb.production()?;
                            Ok(AnyParser::Section(SectionParser::SectionContent(kind, len)))
                        },
                    ));
                }

                SectionParser::SectionContent(kind, length) => {
                    let length = *length as usize;
                    return Ok(match *kind {
                        0x0 => Advancement::YieldTo(
                            window.offset(),
                            AnyParser::Accumulate(Take::new(Accumulator::new(length), length)),
                            |last_state, _| {
                                let AnyParser::Accumulate(acc @ Take { .. }) = last_state else {
                                    unreachable!();
                                };

                                Ok(AnyParser::Section(SectionParser::Done(
                                    SectionType::Custom(acc.production()?),
                                )))
                            },
                        ),

                        0x1 => Advancement::YieldTo(
                            window.offset(),
                            AnyParser::TypeSection(Take::new(TypeParser::default(), length)),
                            |last_state, _| {
                                let AnyParser::TypeSection(ts) = last_state else {
                                    unreachable!();
                                };

                                Ok(AnyParser::Section(SectionParser::Done(SectionType::Type(
                                    ts.production()?,
                                ))))
                            },
                        ),

                        // 0x1 => SectionType::Type(Vec::<Type>::from_wasm_bytes(section)?.1),
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

    fn production(self) -> Result<Self::Production, ParseError> {
        let Self::Done(section_type) = self else {
            unreachable!();
        };

        Ok(section_type)
    }
}
