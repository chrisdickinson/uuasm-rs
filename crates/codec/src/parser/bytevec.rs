use uuasm_nodes::IR;

use crate::{Advancement, Parse, ParseError};

use super::{accumulator::Accumulator, any::AnyParser};

#[derive(Default)]
pub enum ByteVecParser {
    #[default]
    Init,
    Count(u32),
    Ready(Box<[u8]>),
}

impl<T: IR> Parse<T> for ByteVecParser {
    type Production = Box<[u8]>;

    fn advance(
        &mut self,
        _irgen: &mut T,
        window: crate::window::DecodeWindow,
    ) -> crate::ParseResult<T> {
        Ok(match self {
            Self::Init => Advancement::YieldTo(
                window.offset(),
                AnyParser::LEBU32(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::LEBU32(parser) = last_state else {
                        unreachable!()
                    };
                    let items = parser.production(irgen)?;
                    Ok(AnyParser::ByteVec(Self::Count(items)))
                },
            ),
            Self::Count(count) => Advancement::YieldTo(
                window.offset(),
                AnyParser::Accumulate(Accumulator::new(*count as usize)),
                |irgen, last_state, _| {
                    let AnyParser::Accumulate(parser) = last_state else {
                        unreachable!()
                    };
                    let types = parser.production(irgen)?;

                    Ok(AnyParser::ByteVec(Self::Ready(types)))
                },
            ),
            Self::Ready(_) => Advancement::Ready(window.offset()),
        })
    }

    fn production(self, _irgen: &mut T) -> Result<Self::Production, ParseError<<T as IR>::Error>> {
        let Self::Ready(items) = self else {
            unreachable!()
        };
        Ok(items)
    }
}