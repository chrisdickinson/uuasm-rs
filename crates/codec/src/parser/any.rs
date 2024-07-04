use uuasm_nodes::IR;

use crate::{window::DecodeWindow, ExtractTarget, Parse, ParseError, ParseResult};

use super::{
    accumulator::Accumulator, importdescs::ImportDescParser, imports::ImportParser, leb::LEBParser,
    module::ModuleParser, names::NameParser, repeated::Repeated, section::SectionParser,
    types::TypeParser,
};

pub enum AnyParser<T: IR> {
    LEBI32(LEBParser<i32>),
    LEBI64(LEBParser<i64>),
    LEBU32(LEBParser<u32>),
    LEBU64(LEBParser<u64>),
    Accumulate(Accumulator),
    Failed(ParseError),

    Name(NameParser),
    ImportDesc(ImportDescParser),
    Import(ImportParser<T>),
    ImportSection(Repeated<T, ImportParser<T>>),

    Type(TypeParser<T>),
    TypeSection(Repeated<T, TypeParser<T>>),
    Section(SectionParser<T>),
    Module(ModuleParser<T>),
}

macro_rules! repeated_impls {
    ($id:ident) => {
        paste::paste! {
            impl<T: IR> From<[< $id Parser >]<T>> for AnyParser<T> {
                fn from(value: [< $id Parser >]<T>) -> Self {
                    Self::$id(value)
                }
            }

            impl<T: IR> From<Repeated<T, [< $id Parser >]<T>>> for AnyParser<T> {
                fn from(value: Repeated<T, [< $id Parser >]<T>>) -> Self {
                    Self::[< $id Section >](value)
                }
            }

            impl<T: IR> TryFrom<AnyParser<T>> for Repeated<T, [< $id Parser >]<T>> {
                type Error = ParseError;

                fn try_from(value: AnyParser<T>) -> Result<Self, Self::Error> {
                    if let AnyParser::[< $id Section >](v) = value {
                        Ok(v)
                    } else {
                        Err(ParseError::InvalidState(concat!("cannot cast into ", stringify!($id), "Section")))
                    }
                }
            }

            impl<T: IR> TryFrom<AnyParser<T>> for [< $id Parser >]<T> {
                type Error = ParseError;

                fn try_from(value: AnyParser<T>) -> Result<Self, Self::Error> {
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

impl<T: IR> ExtractTarget<AnyProduction<T>> for T::Module {
    fn extract(value: AnyProduction<T>) -> Result<Self, ParseError> {
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
pub enum AnyProduction<T: IR> {
    LEBI32(<LEBParser<i32> as Parse<T>>::Production),
    LEBI64(<LEBParser<i64> as Parse<T>>::Production),
    LEBU32(<LEBParser<u32> as Parse<T>>::Production),
    LEBU64(<LEBParser<u64> as Parse<T>>::Production),

    Name(<NameParser as Parse<T>>::Production),
    ImportDesc(<ImportDescParser as Parse<T>>::Production),
    Import(<ImportParser<T> as Parse<T>>::Production),
    ImportSection(<Repeated<T, ImportParser<T>> as Parse<T>>::Production),

    Type(<TypeParser<T> as Parse<T>>::Production),
    TypeSection(<Repeated<T, TypeParser<T>> as Parse<T>>::Production),
    Accumulate(<Accumulator as Parse<T>>::Production),
    Section(<SectionParser<T> as Parse<T>>::Production),
    Module(<ModuleParser<T> as Parse<T>>::Production),

    Failed(Never),
}

impl<T: IR> Parse<T> for AnyParser<T> {
    type Production = AnyProduction<T>;

    fn advance(&mut self, irgen: &mut T, window: DecodeWindow<'_>) -> ParseResult<T> {
        match self {
            AnyParser::Failed(e) => Err(e.clone()),
            AnyParser::LEBI32(p) => p.advance(irgen, window),
            AnyParser::LEBI64(p) => p.advance(irgen, window),
            AnyParser::LEBU32(p) => p.advance(irgen, window),
            AnyParser::LEBU64(p) => p.advance(irgen, window),
            AnyParser::TypeSection(p) => p.advance(irgen, window),
            AnyParser::Accumulate(p) => p.advance(irgen, window),
            AnyParser::Section(p) => p.advance(irgen, window),
            AnyParser::Module(p) => p.advance(irgen, window),
            AnyParser::Type(p) => p.advance(irgen, window),
            AnyParser::Name(p) => p.advance(irgen, window),
            AnyParser::ImportDesc(p) => p.advance(irgen, window),
            AnyParser::Import(p) => p.advance(irgen, window),
            AnyParser::ImportSection(p) => p.advance(irgen, window),
        }
    }

    fn production(self, irgen: &mut T) -> Result<Self::Production, ParseError> {
        Ok(match self {
            AnyParser::Failed(e) => return Err(e.clone()),
            AnyParser::LEBI32(p) => AnyProduction::LEBI32(p.production(irgen)?),
            AnyParser::LEBI64(p) => AnyProduction::LEBI64(p.production(irgen)?),
            AnyParser::LEBU32(p) => AnyProduction::LEBU32(p.production(irgen)?),
            AnyParser::LEBU64(p) => AnyProduction::LEBU64(p.production(irgen)?),
            AnyParser::TypeSection(p) => AnyProduction::TypeSection(p.production(irgen)?),
            AnyParser::Accumulate(p) => AnyProduction::Accumulate(p.production(irgen)?),
            AnyParser::Section(p) => AnyProduction::Section(p.production(irgen)?),
            AnyParser::Module(p) => AnyProduction::Module(p.production(irgen)?),
            AnyParser::Type(p) => AnyProduction::Type(p.production(irgen)?),
            AnyParser::Name(p) => AnyProduction::Name(p.production(irgen)?),
            AnyParser::ImportDesc(p) => AnyProduction::ImportDesc(p.production(irgen)?),
            AnyParser::Import(p) => AnyProduction::Import(p.production(irgen)?),
            AnyParser::ImportSection(p) => AnyProduction::ImportSection(p.production(irgen)?),
        })
    }
}
