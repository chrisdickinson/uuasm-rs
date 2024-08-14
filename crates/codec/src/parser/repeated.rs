use uuasm_nodes::IR;

use crate::{window::DecodeWindow, Advancement, Parse, ParseErrorKind, ParseResult};

use super::{any::AnyParser, leb::LEBParser};

pub enum Repeated<T: IR, P: Parse<T>> {
    Init,
    Collecting {
        result: Vec<<P as Parse<T>>::Production>,
        expected: usize,
    },
}

impl<T: IR, P: Parse<T>> Repeated<T, P> {
    pub fn times(n: usize) -> Self {
        Self::Collecting {
            result: Vec::with_capacity(n),
            expected: n,
        }
    }
}

impl<T, P> Default for Repeated<T, P>
where
    T: IR,
    P: Parse<T>,
{
    fn default() -> Self {
        Self::Init
    }
}

// The rules are:
// - The inner parser must provide a conversion into AnyParser (to wrap it)
// - The outer repeated parser type must provide a conversion into AnyParser (to wrap it)
// - There must exist a fallible conversion from AnyParser into the inner parser, which allows
//   us to grab the production.
// - There must exist a fallible conversion from AnyParser into the outer parser, so we can extract
//   our state.
impl<T, P> Parse<T> for Repeated<T, P>
where
    T: IR,
    Self: TryFrom<AnyParser<T>, Error = ParseErrorKind<T::Error>>,
    P: Parse<T> + Default + TryFrom<AnyParser<T>, Error = ParseErrorKind<T::Error>>,
    AnyParser<T>: From<P> + From<Self>,
{
    type Production = Box<[P::Production]>;

    fn advance(&mut self, _irgen: &mut T, window: &mut DecodeWindow) -> ParseResult<T> {
        if let Self::Init = self {
            return Ok(Advancement::YieldTo(
                AnyParser::LEBU32(LEBParser::default()),
                |irgen, last_state, _| {
                    let AnyParser::LEBU32(v) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let expected = v.production(irgen)? as usize;
                    Ok(Self::Collecting {
                        result: Vec::with_capacity(expected),
                        expected,
                    }
                    .into())
                },
            ));
        }

        let Self::Collecting { result, expected } = &self else {
            unsafe {
                crate::cold();
                std::hint::unreachable_unchecked()
            }
        };

        if *expected == result.len() {
            return Ok(Advancement::Ready);
        }

        Ok(Advancement::YieldTo(
            P::default().into(),
            |irgen, last_state, this_state| {
                let last_parser: P = last_state.try_into()?;
                let last_production = last_parser.production(irgen)?;
                let Self::Collecting {
                    mut result,
                    expected,
                }: Self = this_state.try_into()?
                else {
                    unsafe {
                        crate::cold();
                        std::hint::unreachable_unchecked()
                    }
                };
                result.push(last_production);
                Ok(Self::Collecting { result, expected }.into())
            },
        ))
    }

    fn production(
        self,
        _irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseErrorKind<T::Error>> {
        let Self::Collecting { result, expected } = self else {
            unsafe {
                crate::cold();
                std::hint::unreachable_unchecked()
            }
        };

        if result.len() != expected {
            return Err(ParseErrorKind::InvalidState(
                "called production() too early on Repeated<P>",
            ));
        }

        Ok(result.into())
    }
}
