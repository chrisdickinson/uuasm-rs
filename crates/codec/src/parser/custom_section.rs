use uuasm_ir::IR;

use crate::{window::DecodeWindow, Advancement, AnyParser, IRError, Parse, ParseErrorKind};

use super::accumulator::Accumulator;

pub enum CustomSectionParser {
    Init(usize),
    Position(usize, usize),
    ReadLen(usize, usize, usize),
    ReadLenAdjust(usize, usize),
    Name(usize, usize, String),
    Ready(String, Box<[u8]>),
}

impl<T: IR> Parse<T> for CustomSectionParser {
    type Production = T::Section;

    fn advance(&mut self, _irgen: &mut T, window: &mut DecodeWindow) -> crate::ParseResult<T> {
        Ok(match self {
            Self::Init(len) => {
                *self = Self::Position(*len, window.position());
                return self.advance(_irgen, window);
            }

            Self::Position(_, _) => Advancement::YieldTo(
                AnyParser::LEBU32(Default::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::LEBU32(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let AnyParser::CustomSection(CustomSectionParser::Position(len, start_pos)) =
                        this_state
                    else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let name_len = parser.production(irgen)?;

                    Ok(AnyParser::CustomSection(CustomSectionParser::ReadLen(
                        len,
                        start_pos,
                        name_len as usize,
                    )))
                },
            ),

            Self::ReadLen(len, start_pos, name_len) => {
                let adjust = window.position() - *start_pos;
                if adjust > *len {
                    return Err(ParseErrorKind::IncompleteCustomSectionName);
                }
                let len = *len - adjust;

                if *name_len > len {
                    return Err(ParseErrorKind::IncompleteCustomSectionName);
                }
                *self = Self::ReadLenAdjust(len, *name_len);
                return self.advance(_irgen, window);
            }

            Self::ReadLenAdjust(_len, name_len) => Advancement::YieldTo(
                AnyParser::Accumulate(Accumulator::new(*name_len)),
                |irgen, last_state, this_state| {
                    let AnyParser::Accumulate(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let AnyParser::CustomSection(CustomSectionParser::ReadLenAdjust(len, name_len)) =
                        this_state
                    else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let name = parser.production(irgen)?;
                    let name = std::str::from_utf8(&name)?;

                    Ok(AnyParser::CustomSection(CustomSectionParser::Name(
                        len,
                        name_len,
                        name.to_string(),
                    )))
                },
            ),

            Self::Name(len, name_len, _) => Advancement::YieldTo(
                AnyParser::Accumulate(Accumulator::new(len.saturating_sub(*name_len))),
                |irgen, last_state, this_state| {
                    let AnyParser::Accumulate(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let AnyParser::CustomSection(CustomSectionParser::Name(_, _, name)) =
                        this_state
                    else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let data = parser.production(irgen)?;
                    Ok(AnyParser::CustomSection(CustomSectionParser::Ready(
                        name, data,
                    )))
                },
            ),

            Self::Ready(_, _) => Advancement::Ready,
        })
    }

    fn production(
        self,
        irgen: &mut T,
    ) -> Result<Self::Production, ParseErrorKind<<T as IR>::Error>> {
        let Self::Ready(name, data) = self else {
            unsafe {
                crate::cold();
                std::hint::unreachable_unchecked()
            };
        };

        Ok(irgen.make_custom_section(name, data).map_err(IRError)?)
    }
}
