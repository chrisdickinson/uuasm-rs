use uuasm_ir::IR;

use crate::{
    parser::{any::AnyParser, leb::LEBParser},
    window::DecodeWindow,
    Advancement, IRError, Parse, ParseErrorKind,
};

use super::accumulator::Accumulator;

#[derive(Default)]
pub enum NameParser {
    #[default]
    Init,
    GotLen(u32),
    Ready(Box<[u8]>),
}

impl<T: IR> Parse<T> for NameParser {
    type Production = <T as IR>::Name;

    fn advance(&mut self, _irgen: &mut T, _window: &mut DecodeWindow) -> crate::ParseResult<T> {
        match self {
            NameParser::Init => Ok(Advancement::YieldTo(
                AnyParser::LEBU32(LEBParser::default()),
                |irgen, last_state, _| {
                    let AnyParser::LEBU32(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        }
                    };

                    Ok(AnyParser::Name(NameParser::GotLen(
                        parser.production(irgen)?,
                    )))
                },
            )),
            NameParser::GotLen(len) => Ok(Advancement::YieldTo(
                AnyParser::Accumulate(Accumulator::new(*len as usize)),
                |irgen, last_state, _| {
                    let AnyParser::Accumulate(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        }
                    };

                    Ok(AnyParser::Name(NameParser::Ready(
                        parser.production(irgen)?,
                    )))
                },
            )),
            NameParser::Ready(_) => Ok(Advancement::Ready),
        }
    }

    fn production(self, irgen: &mut T) -> Result<Self::Production, ParseErrorKind<T::Error>> {
        let Self::Ready(result) = self else {
            unsafe {
                crate::cold();
                std::hint::unreachable_unchecked()
            }
        };

        Ok(irgen.make_name(result).map_err(IRError)?)
    }
}
