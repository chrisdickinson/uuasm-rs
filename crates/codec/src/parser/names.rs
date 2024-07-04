use uuasm_nodes::IR;

use crate::{
    parser::{any::AnyParser, leb::LEBParser},
    window::DecodeWindow,
    Advancement, Parse, ParseError,
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

    fn advance(&mut self, irgen: &mut T, window: DecodeWindow) -> crate::ParseResult<T> {
        match self {
            NameParser::Init => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::LEBU32(LEBParser::default()),
                |irgen, last_state, _| {
                    let AnyParser::LEBU32(parser) = last_state else {
                        unreachable!()
                    };

                    Ok(AnyParser::Name(NameParser::GotLen(
                        parser.production(irgen)?,
                    )))
                },
            )),
            NameParser::GotLen(len) => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::Accumulate(Accumulator::new(*len as usize)),
                |irgen, last_state, _| {
                    let AnyParser::Accumulate(parser) = last_state else {
                        unreachable!()
                    };

                    Ok(AnyParser::Name(NameParser::Ready(
                        parser.production(irgen)?,
                    )))
                },
            )),
            NameParser::Ready(_) => Ok(Advancement::Ready(window.offset())),
        }
    }

    fn production(self, _irgen: &mut T) -> Result<Self::Production, ParseError> {
        let Self::Ready(_result) = self else {
            unreachable!()
        };

        todo!();
        // let string = std::str::from_utf8(&result).map_err(Into::<ParseError>::into)?;
        // Ok(Name(string.to_string()))
    }
}
