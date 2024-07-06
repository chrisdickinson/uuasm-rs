use uuasm_nodes::IR;

use crate::{window::DecodeWindow, ExtractError, ExtractTarget, Parse, ParseError, ParseResult};

use super::{
    accumulator::Accumulator, globaltype::GlobalTypeParser, importdescs::ImportDescParser,
    imports::ImportParser, leb::LEBParser, limits::LimitsParser, module::ModuleParser,
    names::NameParser, repeated::Repeated, section::SectionParser, tabletype::TableTypeParser,
    type_idxs::TypeIdxParser, types::TypeParser,
};

pub enum AnyParser<T: IR> {
    LEBI32(LEBParser<i32>),
    LEBI64(LEBParser<i64>),
    LEBU32(LEBParser<u32>),
    LEBU64(LEBParser<u64>),
    Accumulate(Accumulator),
    Failed(ParseError<T::Error>),

    Name(NameParser),
    TypeIdx(TypeIdxParser<T>),
    Limits(LimitsParser),
    TableType(TableTypeParser<T>),
    GlobalType(GlobalTypeParser<T>),
    ImportDesc(ImportDescParser<T>),
    Import(ImportParser<T>),
    ImportSection(Repeated<T, ImportParser<T>>),
    FunctionSection(Repeated<T, TypeIdxParser<T>>),
    TableSection(Repeated<T, TableTypeParser<T>>),

    Type(TypeParser<T>),
    TypeSection(Repeated<T, TypeParser<T>>),
    Section(SectionParser<T>),
    Module(ModuleParser<T>),
}

macro_rules! repeated_impls {
    ($id:ident, $section:ident) => {
        paste::paste! {
            impl<T: IR> From<[< $id Parser >]<T>> for AnyParser<T> {
                fn from(value: [< $id Parser >]<T>) -> Self {
                    Self::$id(value)
                }
            }

            impl<T: IR> From<Repeated<T, [< $id Parser >]<T>>> for AnyParser<T> {
                fn from(value: Repeated<T, [< $id Parser >]<T>>) -> Self {
                    Self::$section(value)
                }
            }

            impl<T: IR> TryFrom<AnyParser<T>> for Repeated<T, [< $id Parser >]<T>> {
                type Error = ParseError<T::Error>;

                fn try_from(value: AnyParser<T>) -> Result<Self, Self::Error> {
                    if let AnyParser::$section(v) = value {
                        Ok(v)
                    } else {
                        Err(ParseError::InvalidState(concat!("cannot cast into ", stringify!($section))))
                    }
                }
            }

            impl<T: IR> TryFrom<AnyParser<T>> for [< $id Parser >]<T> {
                type Error = ParseError<T::Error>;

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
    ($id:ident) => {
        paste::paste! { repeated_impls!($id, [< $id Section >]); }
    };
}

impl<T: IR> ExtractTarget<AnyProduction<T>> for T::Module {
    fn extract(value: AnyProduction<T>) -> Result<Self, ExtractError> {
        if let AnyProduction::Module(m) = value {
            Ok(m)
        } else {
            Err(ExtractError::Failed)
        }
    }
}

repeated_impls!(Type);
repeated_impls!(Import);
repeated_impls!(TypeIdx, FunctionSection);
repeated_impls!(TableType, TableSection);

pub enum Never {}
pub enum AnyProduction<T: IR> {
    LEBI32(<LEBParser<i32> as Parse<T>>::Production),
    LEBI64(<LEBParser<i64> as Parse<T>>::Production),
    LEBU32(<LEBParser<u32> as Parse<T>>::Production),
    LEBU64(<LEBParser<u64> as Parse<T>>::Production),

    Name(<NameParser as Parse<T>>::Production),
    TypeIdx(<TypeIdxParser<T> as Parse<T>>::Production),
    Limits(<LimitsParser as Parse<T>>::Production),
    TableType(<TableTypeParser<T> as Parse<T>>::Production),
    GlobalType(<GlobalTypeParser<T> as Parse<T>>::Production),
    ImportDesc(<ImportDescParser<T> as Parse<T>>::Production),
    Import(<ImportParser<T> as Parse<T>>::Production),
    ImportSection(<Repeated<T, ImportParser<T>> as Parse<T>>::Production),
    FunctionSection(<Repeated<T, TypeIdxParser<T>> as Parse<T>>::Production),
    TableSection(<Repeated<T, TableTypeParser<T>> as Parse<T>>::Production),

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
            AnyParser::GlobalType(p) => p.advance(irgen, window),
            AnyParser::Limits(p) => p.advance(irgen, window),
            AnyParser::TableType(p) => p.advance(irgen, window),
            AnyParser::TypeIdx(p) => p.advance(irgen, window),
            AnyParser::FunctionSection(p) => p.advance(irgen, window),
            AnyParser::TableSection(p) => p.advance(irgen, window),
        }
    }

    fn production(self, irgen: &mut T) -> Result<Self::Production, ParseError<T::Error>> {
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
            AnyParser::GlobalType(p) => AnyProduction::GlobalType(p.production(irgen)?),
            AnyParser::Limits(p) => AnyProduction::Limits(p.production(irgen)?),
            AnyParser::TableType(p) => AnyProduction::TableType(p.production(irgen)?),
            AnyParser::TypeIdx(p) => AnyProduction::TypeIdx(p.production(irgen)?),
            AnyParser::FunctionSection(p) => AnyProduction::FunctionSection(p.production(irgen)?),
            AnyParser::TableSection(p) => AnyProduction::TableSection(p.production(irgen)?),
        })
    }
}
