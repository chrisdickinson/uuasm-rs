use uuasm_nodes::Module;

use crate::{window::DecodeWindow, Parse, ParseError, ParseResult};

use super::{
    accumulator::Accumulator, leb::LEBParser, module::ModuleParser, section::SectionParser,
    take::Take, types::TypeParser,
};

pub enum AnyParser {
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

impl TryFrom<AnyProduction> for Module {
    type Error = ParseError;

    fn try_from(value: AnyProduction) -> Result<Self, Self::Error> {
        if let AnyProduction::Module(m) = value {
            Ok(m)
        } else {
            Err(ParseError::InvalidProduction)
        }
    }
}

pub enum Never {}
pub enum AnyProduction {
    LEBI32(<LEBParser<i32> as Parse>::Production),
    LEBI64(<LEBParser<i64> as Parse>::Production),
    LEBU32(<LEBParser<u32> as Parse>::Production),
    LEBU64(<LEBParser<u64> as Parse>::Production),
    TypeSection(<Take<TypeParser> as Parse>::Production),
    Accumulate(<Take<Accumulator> as Parse>::Production),
    Section(<SectionParser as Parse>::Production),
    Module(<ModuleParser as Parse>::Production),

    Failed(Never),
}

impl Parse for AnyParser {
    type Production = AnyProduction;

    fn advance(&mut self, window: DecodeWindow<'_>) -> ParseResult {
        match self {
            AnyParser::Failed(e) => Err(e.clone()),
            AnyParser::LEBI32(p) => p.advance(window),
            AnyParser::LEBI64(p) => p.advance(window),
            AnyParser::LEBU32(p) => p.advance(window),
            AnyParser::LEBU64(p) => p.advance(window),
            AnyParser::TypeSection(p) => p.advance(window),
            AnyParser::Accumulate(p) => p.advance(window),
            AnyParser::Section(p) => p.advance(window),
            AnyParser::Module(p) => p.advance(window),
        }
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        Ok(match self {
            AnyParser::Failed(e) => return Err(e.clone()),
            AnyParser::LEBI32(p) => AnyProduction::LEBI32(p.production()?),
            AnyParser::LEBI64(p) => AnyProduction::LEBI64(p.production()?),
            AnyParser::LEBU32(p) => AnyProduction::LEBU32(p.production()?),
            AnyParser::LEBU64(p) => AnyProduction::LEBU64(p.production()?),
            AnyParser::TypeSection(p) => AnyProduction::TypeSection(p.production()?),
            AnyParser::Accumulate(p) => AnyProduction::Accumulate(p.production()?),
            AnyParser::Section(p) => AnyProduction::Section(p.production()?),
            AnyParser::Module(p) => AnyProduction::Module(p.production()?),
        })
    }
}
