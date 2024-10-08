use uuasm_ir::IR;

use crate::{Advancement, IRError, Parse, ParseError};

use super::{accumulator::Accumulator, any::AnyParser};

#[derive(Default)]
pub enum InstrArgSelectParser<T: IR> {
    #[default]
    Init,
    Count(u32),
    Ready(Box<[T::ValType]>),
}

impl<T: IR> Parse<T> for InstrArgSelectParser<T> {
    type Production = Box<[T::ValType]>;

    fn advance(
        &mut self,
        _irgen: &mut T,
        window: &mut DecodeWindow,
    ) -> crate::ParseResult<T> {
        Ok(match self {
            Self::Init => Advancement::YieldTo(
                
                AnyParser::LEBU32(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::LEBU32(parser) = last_state else {
                         unsafe { crate::cold(); std::hint::unreachable_unchecked() }
                    };
                    let items = parser.production(irgen)?;
                    Ok(AnyParser::ArgSelect(Self::Count(items)))
                },
            ),
            Self::Count(count) => Advancement::YieldTo(
                
                AnyParser::Accumulate(Accumulator::new(*count as usize)),
                |irgen, last_state, _| {
                    let AnyParser::Accumulate(parser) = last_state else {
                         unsafe { crate::cold(); std::hint::unreachable_unchecked() }
                    };
                    let types = parser.production(irgen)?;
                    let types: Box<[T::ValType]> = types
                        .iter()
                        .map(|xs| irgen.make_val_type(*xs))
                        .collect::<Result<_, T::Error>>()
                        .map_err(IRError)?;

                    Ok(AnyParser::ArgSelect(Self::Ready(types)))
                },
            ),
            Self::Ready(_) => Advancement::Ready,
        })
    }

    fn production(self, _irgen: &mut T) -> Result<Self::Production, ParseError<<T as IR>::Error>> {
        let Self::Ready(items) = self else {
             unsafe { crate::cold(); std::hint::unreachable_unchecked() }
        };
        Ok(items)
    }
}
