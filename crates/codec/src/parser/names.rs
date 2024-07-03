use uuasm_nodes::Name;

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

impl Parse for NameParser {
    type Production = Name;

    fn advance(&mut self, window: DecodeWindow) -> crate::ParseResult {
        match self {
            NameParser::Init => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::LEBU32(LEBParser::default()),
                |last_state, _| {
                    let AnyParser::LEBU32(parser) = last_state else {
                        unreachable!()
                    };

                    Ok(AnyParser::Name(NameParser::GotLen(parser.production()?)))
                },
            )),
            NameParser::GotLen(len) => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::Accumulate(Accumulator::new(*len as usize)),
                |last_state, _| {
                    let AnyParser::Accumulate(parser) = last_state else {
                        unreachable!()
                    };

                    Ok(AnyParser::Name(NameParser::Ready(parser.production()?)))
                },
            )),
            NameParser::Ready(_) => Ok(Advancement::Ready(window.offset())),
        }
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        let Self::Ready(result) = self else {
            unreachable!()
        };

        let string = std::str::from_utf8(&result).map_err(Into::<ParseError>::into)?;
        Ok(Name(string.to_string()))
    }
}
