use crate::{window::DecodeWindow, Advancement, Parse, ParseError, ParseResult};

use super::{any::AnyParser, leb::LEBParser};

pub enum Repeated<P: Parse> {
    Init,
    Collecting {
        result: Vec<P::Production>,
        expected: usize,
    },
}

impl<P> Default for Repeated<P>
where
    P: Parse,
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
impl<P> Parse for Repeated<P>
where
    Self: TryFrom<AnyParser, Error = ParseError>,
    P: Parse + Default + TryFrom<AnyParser, Error = ParseError>,
    AnyParser: From<P> + From<Self>,
{
    type Production = Box<[P::Production]>;

    fn advance(&mut self, window: DecodeWindow) -> ParseResult {
        if let Self::Init = self {
            return Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::LEBU32(LEBParser::default()),
                |last_state, _| {
                    let AnyParser::LEBU32(v) = last_state else {
                        unreachable!();
                    };
                    let expected = v.production()? as usize;
                    Ok(Self::Collecting {
                        result: Vec::with_capacity(expected),
                        expected,
                    }
                    .into())
                },
            ));
        }

        let Self::Collecting { result, expected } = &self else {
            unreachable!()
        };

        if *expected == result.len() {
            return Ok(Advancement::Ready(window.offset()));
        }

        Ok(Advancement::YieldTo(
            window.offset(),
            P::default().into(),
            |last_state, this_state| {
                let last_parser: P = last_state.try_into()?;
                let last_production = last_parser.production()?;
                let Self::Collecting {
                    mut result,
                    expected,
                }: Self = this_state.try_into()?
                else {
                    unreachable!()
                };
                result.push(last_production);
                Ok(Self::Collecting { result, expected }.into())
            },
        ))
    }

    fn production(self) -> Result<Self::Production, crate::ParseError> {
        let Self::Collecting { result, expected } = self else {
            unreachable!()
        };

        if result.len() != expected {
            return Err(ParseError::InvalidState(
                "called production() too early on Repeated<P>",
            ));
        }

        Ok(result.into())
    }
}
