use uuasm_nodes::IR;

use crate::{window::DecodeWindow, Advancement, IRError, Parse, ParseError, ParseResult};

use super::any::AnyParser;

#[derive(Default)]
pub enum MemTypeParser<T: IR> {
    #[default]
    Init,
    Ready(T::Limits),
}

impl<T: IR> Parse<T> for MemTypeParser<T> {
    type Production = T::MemType;

    fn advance(&mut self, _irgen: &mut T, window: DecodeWindow) -> ParseResult<T> {
        match self {
            Self::Init => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::Limits(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::Limits(parser) = last_state else {
                         unsafe { crate::cold(); std::hint::unreachable_unchecked() }
                    };
                    let limits = parser.production(irgen)?;

                    Ok(AnyParser::MemType(Self::Ready(limits)))
                },
            )),
            Self::Ready(_) => Ok(Advancement::Ready(window.offset())),
        }
    }

    fn production(self, irgen: &mut T) -> Result<Self::Production, ParseError<<T as IR>::Error>> {
        let Self::Ready(production) = self else {
             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
        };
        Ok(irgen.make_mem_type(production).map_err(IRError)?)
    }
}
