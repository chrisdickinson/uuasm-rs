use uuasm_nodes::Module;

use crate::{window::DecodeWindow, Parse, ParseError, ParseResult};

use super::{
    accumulator::Accumulator, leb::LEBParser, module::ModuleParser, section::SectionParser,
    take::Take, types::TypeParser,
};

pub(crate) enum ParseState {
    LEBI32(LEBParser<i32>),
    LEBI64(LEBParser<i64>),
    LEBU32(LEBParser<u32>),
    LEBU64(LEBParser<u64>),
    TypeSection(Take<TypeParser>),
    Accumulate(Take<Accumulator>),
    Section(SectionParser),
    Module(ModuleParser),

    Failed(ParseError),
}

impl Parse for ParseState {
    type Production = Module;

    fn advance(&mut self, window: DecodeWindow<'_>) -> ParseResult {
        match self {
            ParseState::Failed(e) => Err(e.clone()),
            ParseState::LEBI32(p) => p.advance(window),
            ParseState::LEBI64(p) => p.advance(window),
            ParseState::LEBU32(p) => p.advance(window),
            ParseState::LEBU64(p) => p.advance(window),
            ParseState::TypeSection(p) => p.advance(window),
            ParseState::Accumulate(p) => p.advance(window),
            ParseState::Section(p) => p.advance(window),
            ParseState::Module(p) => p.advance(window),
        }
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        let Self::Module(module) = self else {
            unreachable!();
        };

        module.production()
    }
}
