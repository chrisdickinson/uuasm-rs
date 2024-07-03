use uuasm_nodes::Module;

use crate::{window::DecodeWindow, Parse, ParseError, ParseResult};

use super::{
    accumulator::Accumulator, importdescs::ImportDescParser, imports::ImportParser, leb::LEBParser,
    module::ModuleParser, names::NameParser, repeated::Repeated, section::SectionParser,
    types::TypeParser,
};

pub enum AnyParser {
    Name(NameParser),
    LEBI32(LEBParser<i32>),
    LEBI64(LEBParser<i64>),
    LEBU32(LEBParser<u32>),
    LEBU64(LEBParser<u64>),

    ImportDesc(ImportDescParser),
    Import(ImportParser),
    ImportSection(Repeated<ImportParser>),

    Type(TypeParser),
    TypeSection(Repeated<TypeParser>),
    Accumulate(Accumulator),
    Section(SectionParser),
    Module(ModuleParser),

    Failed(ParseError),
}

macro_rules! repeated_impls {
    ($id:ident) => {
        paste::paste! {
            impl From<[< $id Parser >]> for AnyParser {
                fn from(value: [< $id Parser >]) -> Self {
                    Self::$id(value)
                }
            }

            impl From<Repeated<[< $id Parser >]>> for AnyParser {
                fn from(value: Repeated<[< $id Parser >]>) -> Self {
                    Self::[< $id Section >](value)
                }
            }

            impl TryFrom<AnyParser> for Repeated<[< $id Parser >]> {
                type Error = ParseError;

                fn try_from(value: AnyParser) -> Result<Self, Self::Error> {
                    if let AnyParser::[< $id Section >](v) = value {
                        Ok(v)
                    } else {
                        Err(ParseError::InvalidState(concat!("cannot cast into ", stringify!($id), "Section")))
                    }
                }
            }

            impl TryFrom<AnyParser> for [< $id Parser >] {
                type Error = ParseError;

                fn try_from(value: AnyParser) -> Result<Self, Self::Error> {
                    if let AnyParser::$id(parser) = value {
                        Ok(parser)
                    } else {
                        Err(ParseError::InvalidState(concat!("expected AnyParser::", stringify!($id))))
                    }
                }
            }
        }
    };
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

repeated_impls!(Type);
repeated_impls!(Import);

pub enum Never {}
pub enum AnyProduction {
    LEBI32(<LEBParser<i32> as Parse>::Production),
    LEBI64(<LEBParser<i64> as Parse>::Production),
    LEBU32(<LEBParser<u32> as Parse>::Production),
    LEBU64(<LEBParser<u64> as Parse>::Production),

    Name(<NameParser as Parse>::Production),
    ImportDesc(<ImportDescParser as Parse>::Production),
    Import(<ImportParser as Parse>::Production),
    ImportSection(<Repeated<ImportParser> as Parse>::Production),

    Type(<TypeParser as Parse>::Production),
    TypeSection(<Repeated<TypeParser> as Parse>::Production),
    Accumulate(<Accumulator as Parse>::Production),
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
            AnyParser::Type(p) => p.advance(window),
            AnyParser::Name(p) => p.advance(window),
            AnyParser::ImportDesc(p) => p.advance(window),
            AnyParser::Import(p) => p.advance(window),
            AnyParser::ImportSection(p) => p.advance(window),
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
            AnyParser::Type(p) => AnyProduction::Type(p.production()?),
            AnyParser::Name(p) => AnyProduction::Name(p.production()?),
            AnyParser::ImportDesc(p) => AnyProduction::ImportDesc(p.production()?),
            AnyParser::Import(p) => AnyProduction::Import(p.production()?),
            AnyParser::ImportSection(p) => AnyProduction::ImportSection(p.production()?),
        })
    }
}
