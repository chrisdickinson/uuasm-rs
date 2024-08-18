use uuasm_ir::IR;

use crate::{window::DecodeWindow, Advancement, IRError, Parse, ParseErrorKind};

use super::any::AnyParser;

#[derive(Default)]
pub enum LimitsParser {
    #[default]
    Init,
    LowerBound,
    Bounded,
    GotLowerBound(u32),
    Ready(u32, Option<u32>),
}

impl<T: IR> Parse<T> for LimitsParser {
    type Production = T::Limits;

    fn advance(&mut self, _irgen: &mut T, window: &mut DecodeWindow) -> crate::ParseResult<T> {
        match self {
            Self::Init => {
                *self = match window.take()? {
                    0x00 => Self::LowerBound,
                    0x01 => Self::Bounded,
                    unk => return Err(ParseErrorKind::BadLimitType(unk)),
                };
                self.advance(_irgen, window)
            }

            Self::LowerBound => Ok(Advancement::YieldTo(
                AnyParser::LEBU32(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::LEBU32(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let bound = parser.production(irgen)?;
                    Ok(AnyParser::Limits(Self::Ready(bound, None)))
                },
            )),

            Self::Bounded => Ok(Advancement::YieldTo(
                AnyParser::LEBU32(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::LEBU32(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let bound = parser.production(irgen)?;
                    Ok(AnyParser::Limits(Self::GotLowerBound(bound)))
                },
            )),

            Self::GotLowerBound(_) => Ok(Advancement::YieldTo(
                AnyParser::LEBU32(Default::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::LEBU32(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let AnyParser::Limits(Self::GotLowerBound(lower)) = this_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let bound = parser.production(irgen)?;
                    Ok(AnyParser::Limits(Self::Ready(lower, Some(bound))))
                },
            )),
            Self::Ready(_, _) => Ok(Advancement::Ready),
        }
    }

    fn production(
        self,
        irgen: &mut T,
    ) -> Result<Self::Production, ParseErrorKind<<T as IR>::Error>> {
        let Self::Ready(lower, upper) = self else {
            unsafe {
                crate::cold();
                std::hint::unreachable_unchecked()
            };
        };

        Ok(irgen.make_limits(lower, upper).map_err(IRError)?)
    }
}
