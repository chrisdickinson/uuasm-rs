use uuasm_nodes::IR;

use crate::{window::DecodeWindow, Advancement, Parse, ParseErrorKind};

use super::any::AnyParser;

#[derive(Default)]
pub enum InstrArgTableParser {
    #[default]
    Init,
    Items(Box<[u32]>),
    Ready(Box<[u32]>, u32),
}

impl<T: IR> Parse<T> for InstrArgTableParser {
    type Production = (Box<[u32]>, u32);

    fn advance(&mut self, _irgen: &mut T, window: &mut DecodeWindow) -> crate::ParseResult<T> {
        Ok(match self {
            Self::Init => Advancement::YieldTo(
                AnyParser::RepeatedLEBU32(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::RepeatedLEBU32(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        }
                    };
                    let items = parser.production(irgen)?;
                    Ok(AnyParser::ArgTable(Self::Items(items)))
                },
            ),
            Self::Items(_) => Advancement::YieldTo(
                AnyParser::LEBU32(Default::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::LEBU32(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        }
                    };
                    let AnyParser::ArgTable(Self::Items(items)) = this_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        }
                    };
                    let alternate = parser.production(irgen)?;
                    Ok(AnyParser::ArgTable(Self::Ready(items, alternate)))
                },
            ),
            Self::Ready(_, _) => Advancement::Ready,
        })
    }

    fn production(
        self,
        _irgen: &mut T,
    ) -> Result<Self::Production, ParseErrorKind<<T as IR>::Error>> {
        let Self::Ready(items, alternate) = self else {
            unsafe {
                crate::cold();
                std::hint::unreachable_unchecked()
            }
        };
        Ok((items, alternate))
    }
}
