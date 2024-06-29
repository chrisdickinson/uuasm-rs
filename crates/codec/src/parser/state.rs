use uuasm_nodes::Module;

use crate::{window::DecodeWindow, Parse, ParseError, ParseResult};

use super::{
    accumulator::Accumulator, leb::LEBParser, module::ModuleParser, section::SectionParser,
    take::Take, types::TypeParser,
};

pub enum AnyState {
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

impl TryFrom<AnyStateProduction> for Module {
    type Error = ParseError;

    fn try_from(value: AnyStateProduction) -> Result<Self, Self::Error> {
        if let AnyStateProduction::Module(m) = value {
            Ok(m)
        } else {
            Err(ParseError::InvalidProduction)
        }
    }
}

pub enum Never {}
pub enum AnyStateProduction {
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

impl Parse for AnyState {
    type Production = AnyStateProduction;

    fn advance(&mut self, window: DecodeWindow<'_>) -> ParseResult {
        match self {
            AnyState::Failed(e) => Err(e.clone()),
            AnyState::LEBI32(p) => p.advance(window),
            AnyState::LEBI64(p) => p.advance(window),
            AnyState::LEBU32(p) => p.advance(window),
            AnyState::LEBU64(p) => p.advance(window),
            AnyState::TypeSection(p) => p.advance(window),
            AnyState::Accumulate(p) => p.advance(window),
            AnyState::Section(p) => p.advance(window),
            AnyState::Module(p) => p.advance(window),
        }
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        Ok(match self {
            AnyState::Failed(e) => return Err(e.clone()),
            AnyState::LEBI32(p) => AnyStateProduction::LEBI32(p.production()?),
            AnyState::LEBI64(p) => AnyStateProduction::LEBI64(p.production()?),
            AnyState::LEBU32(p) => AnyStateProduction::LEBU32(p.production()?),
            AnyState::LEBU64(p) => AnyStateProduction::LEBU64(p.production()?),
            AnyState::TypeSection(p) => AnyStateProduction::TypeSection(p.production()?),
            AnyState::Accumulate(p) => AnyStateProduction::Accumulate(p.production()?),
            AnyState::Section(p) => AnyStateProduction::Section(p.production()?),
            AnyState::Module(p) => AnyStateProduction::Module(p.production()?),
        })
    }
}
