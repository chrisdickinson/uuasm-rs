use uuasm_ir::IR;

use crate::{
    window::DecodeWindow, ExtractError, ExtractTarget, Parse, ParseErrorKind, ParseResult,
};

use super::{
    accumulator::Accumulator, block::BlockParser, block_ifelse::IfElseBlockParser,
    blocktype::BlockTypeParser, bytevec::ByteVecParser, code::CodeParser,
    custom_section::CustomSectionParser, data::DataParser, elem::ElemParser, export::ExportParser,
    exportdesc::ExportDescParser, expr::ExprParser, func::FuncParser, func_idxs::FuncIdxParser,
    global::GlobalParser, global_idxs::GlobalIdxParser, globaltype::GlobalTypeParser,
    importdescs::ImportDescParser, imports::ImportParser,
    instrarg_multibyte::InstrArgMultibyteParser, instrarg_refnull::InstrArgRefNullParser,
    instrarg_table::InstrArgTableParser, leb::LEBParser, limits::LimitsParser, local::LocalParser,
    mem_idxs::MemIdxParser, memtype::MemTypeParser, module::ModuleParser, names::NameParser,
    repeated::Repeated, section::SectionParser, table_idxs::TableIdxParser,
    tabletype::TableTypeParser, type_idxs::TypeIdxParser, types::TypeParser,
};

pub enum AnyParser<T: IR> {
    LEBI32(LEBParser<i32>),
    LEBI64(LEBParser<i64>),
    LEBU32(LEBParser<u32>),
    LEBU64(LEBParser<u64>),
    CustomSection(CustomSectionParser),
    RepeatedLEBU32(Repeated<T, LEBParser<u32>>),
    ByteVec(ByteVecParser),
    ArgMultibyte(InstrArgMultibyteParser),
    ArgTable(InstrArgTableParser),
    ArgRefNull(InstrArgRefNullParser),
    Accumulate(Accumulator),
    Failed(ParseErrorKind<T::Error>),

    IfElseBlock(IfElseBlockParser<T>),
    Block(BlockParser<T>),
    BlockType(BlockTypeParser<T>),

    TypeIdx(TypeIdxParser<T>),
    GlobalIdx(GlobalIdxParser<T>),
    MemIdx(MemIdxParser<T>),
    TableIdx(TableIdxParser<T>),
    FuncIdx(FuncIdxParser<T>),

    Code(CodeParser<T>),
    Data(DataParser<T>),
    Elem(ElemParser<T>),
    Export(ExportParser<T>),
    ExportDesc(ExportDescParser<T>),
    Expr(ExprParser<T>),
    LocalList(Repeated<T, LocalParser<T>>),
    ExprList(Repeated<T, ExprParser<T>>),
    Func(FuncParser<T>),
    Global(GlobalParser<T>),
    Local(LocalParser<T>),
    Name(NameParser),
    Limits(LimitsParser),
    MemType(MemTypeParser<T>),
    TableType(TableTypeParser<T>),
    GlobalType(GlobalTypeParser<T>),
    ImportDesc(ImportDescParser<T>),
    Import(ImportParser<T>),
    ImportSection(Repeated<T, ImportParser<T>>),
    FunctionSection(Repeated<T, TypeIdxParser<T>>),
    TableSection(Repeated<T, TableTypeParser<T>>),
    ExportSection(Repeated<T, ExportParser<T>>),
    MemorySection(Repeated<T, MemTypeParser<T>>),
    GlobalSection(Repeated<T, GlobalParser<T>>),
    CodeSection(Repeated<T, CodeParser<T>>),
    DataSection(Repeated<T, DataParser<T>>),
    ElementSection(Repeated<T, ElemParser<T>>),

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
                type Error = ParseErrorKind<T::Error>;

                fn try_from(value: AnyParser<T>) -> Result<Self, Self::Error> {
                    if let AnyParser::$section(v) = value {
                        Ok(v)
                    } else {
                        Err(ParseErrorKind::InvalidState(concat!("cannot cast into ", stringify!($section))))
                    }
                }
            }

            impl<T: IR> TryFrom<AnyParser<T>> for [< $id Parser >]<T> {
                type Error = ParseErrorKind<T::Error>;

                fn try_from(value: AnyParser<T>) -> Result<Self, Self::Error> {
                    if let AnyParser::$id(parser) = value {
                        Ok(parser)
                    } else {
                        Err(ParseErrorKind::InvalidState(concat!("expected AnyParser::", stringify!($id))))
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
repeated_impls!(Local, LocalList);
repeated_impls!(Expr, ExprList);
repeated_impls!(Export);
repeated_impls!(Global);
repeated_impls!(Code);
repeated_impls!(Data);
repeated_impls!(Elem, ElementSection);
repeated_impls!(MemType, MemorySection);

impl<T: IR> From<LEBParser<u32>> for AnyParser<T> {
    fn from(value: LEBParser<u32>) -> Self {
        Self::LEBU32(value)
    }
}

impl<T: IR> From<Repeated<T, LEBParser<u32>>> for AnyParser<T> {
    fn from(value: Repeated<T, LEBParser<u32>>) -> Self {
        Self::RepeatedLEBU32(value)
    }
}

impl<T: IR> TryFrom<AnyParser<T>> for Repeated<T, LEBParser<u32>> {
    type Error = ParseErrorKind<T::Error>;

    fn try_from(value: AnyParser<T>) -> Result<Self, Self::Error> {
        if let AnyParser::RepeatedLEBU32(v) = value {
            Ok(v)
        } else {
            Err(ParseErrorKind::InvalidState(concat!(
                "cannot cast into ",
                stringify!($section)
            )))
        }
    }
}

impl<T: IR> TryFrom<AnyParser<T>> for LEBParser<u32> {
    type Error = ParseErrorKind<T::Error>;

    fn try_from(value: AnyParser<T>) -> Result<Self, Self::Error> {
        if let AnyParser::LEBU32(parser) = value {
            Ok(parser)
        } else {
            Err(ParseErrorKind::InvalidState(concat!(
                "expected AnyParser::",
                stringify!($id)
            )))
        }
    }
}

pub enum Never {}
pub enum AnyProduction<T: IR> {
    LEBI32(<LEBParser<i32> as Parse<T>>::Production),
    LEBI64(<LEBParser<i64> as Parse<T>>::Production),
    LEBU32(<LEBParser<u32> as Parse<T>>::Production),
    LEBU64(<LEBParser<u64> as Parse<T>>::Production),
    ByteVec(<ByteVecParser as Parse<T>>::Production),

    RepeatedLEBU32(<Repeated<T, LEBParser<u32>> as Parse<T>>::Production),
    ArgMultibyte(<InstrArgMultibyteParser as Parse<T>>::Production),
    ArgTable(<InstrArgTableParser as Parse<T>>::Production),
    ArgRefNull(<InstrArgRefNullParser as Parse<T>>::Production),

    TypeIdx(<TypeIdxParser<T> as Parse<T>>::Production),
    FuncIdx(<FuncIdxParser<T> as Parse<T>>::Production),
    MemIdx(<MemIdxParser<T> as Parse<T>>::Production),
    GlobalIdx(<GlobalIdxParser<T> as Parse<T>>::Production),
    TableIdx(<TableIdxParser<T> as Parse<T>>::Production),

    Block(<BlockParser<T> as Parse<T>>::Production),
    IfElseBlock(<IfElseBlockParser<T> as Parse<T>>::Production),
    BlockType(<BlockTypeParser<T> as Parse<T>>::Production),
    Code(<CodeParser<T> as Parse<T>>::Production),
    Data(<DataParser<T> as Parse<T>>::Production),
    Elem(<ElemParser<T> as Parse<T>>::Production),
    Export(<ExportParser<T> as Parse<T>>::Production),
    ExportDesc(<ExportDescParser<T> as Parse<T>>::Production),
    ExportSection(<Repeated<T, ExportParser<T>> as Parse<T>>::Production),
    CustomSection(<CustomSectionParser as Parse<T>>::Production),
    Expr(<ExprParser<T> as Parse<T>>::Production),
    Func(<FuncParser<T> as Parse<T>>::Production),
    Global(<GlobalParser<T> as Parse<T>>::Production),
    Local(<LocalParser<T> as Parse<T>>::Production),
    LocalList(<Repeated<T, LocalParser<T>> as Parse<T>>::Production),
    ExprList(<Repeated<T, ExprParser<T>> as Parse<T>>::Production),

    Name(<NameParser as Parse<T>>::Production),
    Limits(<LimitsParser as Parse<T>>::Production),
    MemType(<MemTypeParser<T> as Parse<T>>::Production),
    TableType(<TableTypeParser<T> as Parse<T>>::Production),
    GlobalType(<GlobalTypeParser<T> as Parse<T>>::Production),
    ImportDesc(<ImportDescParser<T> as Parse<T>>::Production),
    Import(<ImportParser<T> as Parse<T>>::Production),
    ImportSection(<Repeated<T, ImportParser<T>> as Parse<T>>::Production),
    FunctionSection(<Repeated<T, TypeIdxParser<T>> as Parse<T>>::Production),
    TableSection(<Repeated<T, TableTypeParser<T>> as Parse<T>>::Production),
    DataSection(<Repeated<T, DataParser<T>> as Parse<T>>::Production),
    ElementSection(<Repeated<T, ElemParser<T>> as Parse<T>>::Production),
    MemorySection(<Repeated<T, MemTypeParser<T>> as Parse<T>>::Production),
    GlobalSection(<Repeated<T, GlobalParser<T>> as Parse<T>>::Production),
    CodeSection(<Repeated<T, CodeParser<T>> as Parse<T>>::Production),

    Type(<TypeParser<T> as Parse<T>>::Production),
    TypeSection(<Repeated<T, TypeParser<T>> as Parse<T>>::Production),
    Accumulate(<Accumulator as Parse<T>>::Production),
    Section(<SectionParser<T> as Parse<T>>::Production),
    Module(<ModuleParser<T> as Parse<T>>::Production),

    Failed(Never),
}

impl<T: IR> Parse<T> for AnyParser<T> {
    type Production = AnyProduction<T>;

    #[inline]
    fn advance(&mut self, irgen: &mut T, window: &mut DecodeWindow<'_>) -> ParseResult<T> {
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
            AnyParser::Code(p) => p.advance(irgen, window),
            AnyParser::Data(p) => p.advance(irgen, window),
            AnyParser::Elem(p) => p.advance(irgen, window),
            AnyParser::Export(p) => p.advance(irgen, window),
            AnyParser::Expr(p) => p.advance(irgen, window),
            AnyParser::Func(p) => p.advance(irgen, window),
            AnyParser::Global(p) => p.advance(irgen, window),
            AnyParser::Local(p) => p.advance(irgen, window),
            AnyParser::ExportDesc(p) => p.advance(irgen, window),
            AnyParser::ExportSection(p) => p.advance(irgen, window),
            AnyParser::GlobalIdx(p) => p.advance(irgen, window),
            AnyParser::MemIdx(p) => p.advance(irgen, window),
            AnyParser::TableIdx(p) => p.advance(irgen, window),
            AnyParser::FuncIdx(p) => p.advance(irgen, window),
            AnyParser::LocalList(p) => p.advance(irgen, window),
            AnyParser::MemType(p) => p.advance(irgen, window),
            AnyParser::MemorySection(p) => p.advance(irgen, window),
            AnyParser::GlobalSection(p) => p.advance(irgen, window),
            AnyParser::CodeSection(p) => p.advance(irgen, window),
            AnyParser::DataSection(p) => p.advance(irgen, window),
            AnyParser::ElementSection(p) => p.advance(irgen, window),
            AnyParser::RepeatedLEBU32(p) => p.advance(irgen, window),
            AnyParser::IfElseBlock(p) => p.advance(irgen, window),
            AnyParser::Block(p) => p.advance(irgen, window),
            AnyParser::BlockType(p) => p.advance(irgen, window),
            AnyParser::ArgMultibyte(p) => p.advance(irgen, window),
            AnyParser::ArgTable(p) => p.advance(irgen, window),
            AnyParser::ArgRefNull(p) => p.advance(irgen, window),
            AnyParser::ByteVec(p) => p.advance(irgen, window),
            AnyParser::ExprList(p) => p.advance(irgen, window),
            AnyParser::CustomSection(p) => p.advance(irgen, window),
        }
    }

    fn production(self, irgen: &mut T) -> Result<Self::Production, ParseErrorKind<T::Error>> {
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
            AnyParser::Code(p) => AnyProduction::Code(p.production(irgen)?),
            AnyParser::Data(p) => AnyProduction::Data(p.production(irgen)?),
            AnyParser::Elem(p) => AnyProduction::Elem(p.production(irgen)?),
            AnyParser::Export(p) => AnyProduction::Export(p.production(irgen)?),
            AnyParser::Expr(p) => AnyProduction::Expr(p.production(irgen)?),
            AnyParser::Func(p) => AnyProduction::Func(p.production(irgen)?),
            AnyParser::Global(p) => AnyProduction::Global(p.production(irgen)?),
            AnyParser::Local(p) => AnyProduction::Local(p.production(irgen)?),
            AnyParser::LocalList(p) => AnyProduction::LocalList(p.production(irgen)?),
            AnyParser::ExportDesc(p) => AnyProduction::ExportDesc(p.production(irgen)?),
            AnyParser::GlobalIdx(p) => AnyProduction::GlobalIdx(p.production(irgen)?),
            AnyParser::MemIdx(p) => AnyProduction::MemIdx(p.production(irgen)?),
            AnyParser::TableIdx(p) => AnyProduction::TableIdx(p.production(irgen)?),
            AnyParser::FuncIdx(p) => AnyProduction::FuncIdx(p.production(irgen)?),
            AnyParser::MemType(p) => AnyProduction::MemType(p.production(irgen)?),
            AnyParser::ExportSection(p) => AnyProduction::ExportSection(p.production(irgen)?),
            AnyParser::DataSection(p) => AnyProduction::DataSection(p.production(irgen)?),
            AnyParser::ElementSection(p) => AnyProduction::ElementSection(p.production(irgen)?),
            AnyParser::CodeSection(p) => AnyProduction::CodeSection(p.production(irgen)?),
            AnyParser::MemorySection(p) => AnyProduction::MemorySection(p.production(irgen)?),
            AnyParser::GlobalSection(p) => AnyProduction::GlobalSection(p.production(irgen)?),
            AnyParser::RepeatedLEBU32(p) => AnyProduction::RepeatedLEBU32(p.production(irgen)?),
            AnyParser::IfElseBlock(p) => AnyProduction::IfElseBlock(p.production(irgen)?),
            AnyParser::Block(p) => AnyProduction::Block(p.production(irgen)?),
            AnyParser::BlockType(p) => AnyProduction::BlockType(p.production(irgen)?),
            AnyParser::ArgMultibyte(p) => AnyProduction::ArgMultibyte(p.production(irgen)?),
            AnyParser::ArgTable(p) => AnyProduction::ArgTable(p.production(irgen)?),
            AnyParser::ArgRefNull(p) => AnyProduction::ArgRefNull(p.production(irgen)?),
            AnyParser::ByteVec(p) => AnyProduction::ByteVec(p.production(irgen)?),
            AnyParser::ExprList(p) => AnyProduction::ExprList(p.production(irgen)?),
            AnyParser::CustomSection(p) => AnyProduction::CustomSection(p.production(irgen)?),
        })
    }
}

impl<T: IR> std::fmt::Debug for AnyParser<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LEBI32(_) => f.debug_tuple("LEBI32").finish(),
            Self::LEBI64(_) => f.debug_tuple("LEBI64").finish(),
            Self::LEBU32(_) => f.debug_tuple("LEBU32").finish(),
            Self::LEBU64(_) => f.debug_tuple("LEBU64").finish(),
            Self::RepeatedLEBU32(_) => f.debug_tuple("RepeatedLEBU32").finish(),
            Self::ArgMultibyte(_) => f.debug_tuple("ArgMultibyte").finish(),
            Self::ArgTable(_) => f.debug_tuple("ArgTable").finish(),
            Self::ByteVec(_) => f.debug_tuple("ByteVec").finish(),
            Self::ArgRefNull(_) => f.debug_tuple("ArgRefNull").finish(),
            Self::Accumulate(_) => f.debug_tuple("Accumulate").finish(),
            Self::Failed(_) => f.debug_tuple("Failed").finish(),
            Self::TypeIdx(_) => f.debug_tuple("TypeIdx").finish(),
            Self::GlobalIdx(_) => f.debug_tuple("GlobalIdx").finish(),
            Self::MemIdx(_) => f.debug_tuple("MemIdx").finish(),
            Self::TableIdx(_) => f.debug_tuple("TableIdx").finish(),
            Self::FuncIdx(_) => f.debug_tuple("FuncIdx").finish(),
            Self::Block(_) => f.debug_tuple("Block").finish(),
            Self::IfElseBlock(_) => f.debug_tuple("IfElseBlock").finish(),
            Self::BlockType(_) => f.debug_tuple("BlockType").finish(),
            Self::Code(_) => f.debug_tuple("Code").finish(),
            Self::Data(_) => f.debug_tuple("Data").finish(),
            Self::Elem(_) => f.debug_tuple("Elem").finish(),
            Self::Export(_) => f.debug_tuple("Export").finish(),
            Self::ExportDesc(_) => f.debug_tuple("ExportDesc").finish(),
            Self::Expr(_) => f.debug_tuple("Expr").finish(),
            Self::LocalList(_) => f.debug_tuple("LocalList").finish(),
            Self::ExprList(_) => f.debug_tuple("ExprList").finish(),
            Self::Func(_) => f.debug_tuple("Func").finish(),
            Self::Global(_) => f.debug_tuple("Global").finish(),
            Self::Local(_) => f.debug_tuple("Local").finish(),
            Self::Name(_) => f.debug_tuple("Name").finish(),
            Self::Limits(_) => f.debug_tuple("Limits").finish(),
            Self::MemType(_) => f.debug_tuple("MemType").finish(),
            Self::TableType(_) => f.debug_tuple("TableType").finish(),
            Self::GlobalType(_) => f.debug_tuple("GlobalType").finish(),
            Self::ImportDesc(_) => f.debug_tuple("ImportDesc").finish(),
            Self::Import(_) => f.debug_tuple("Import").finish(),
            Self::ImportSection(_) => f.debug_tuple("ImportSection").finish(),
            Self::FunctionSection(_) => f.debug_tuple("FunctionSection").finish(),
            Self::TableSection(_) => f.debug_tuple("TableSection").finish(),
            Self::ExportSection(_) => f.debug_tuple("ExportSection").finish(),
            Self::MemorySection(_) => f.debug_tuple("MemorySection").finish(),
            Self::GlobalSection(_) => f.debug_tuple("GlobalSection").finish(),
            Self::CodeSection(_) => f.debug_tuple("CodeSection").finish(),
            Self::DataSection(_) => f.debug_tuple("DataSection").finish(),
            Self::ElementSection(_) => f.debug_tuple("ElementSection").finish(),
            Self::CustomSection(_) => f.debug_tuple("CustomSection").finish(),
            Self::Type(_) => f.debug_tuple("Type").finish(),
            Self::TypeSection(_) => f.debug_tuple("TypeSection").finish(),
            Self::Section(_) => f.debug_tuple("Section").finish(),
            Self::Module(_) => f.debug_tuple("Module").finish(),
        }
    }
}
