#![allow(dead_code)]
use std::{error::Error, fmt::Debug};

use thiserror::Error;

#[derive(Debug, PartialEq, Clone)]
pub struct ByteVec(pub Box<[u8]>);
#[derive(Debug, PartialEq, Clone)]
pub struct Name(pub String);

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum NumType {
    I32,
    I64,
    F32,
    F64,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum VecType {
    V128,
}

#[derive(Debug, PartialEq, Clone, Copy, Default)]
pub enum RefType {
    #[default]
    FuncRef,
    ExternRef,
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ValType {
    NumType(NumType),
    VecType(VecType),
    RefType(RefType),
}

#[derive(Debug, PartialEq, Clone, Default)]
pub struct ResultType(pub Box<[ValType]>);

#[derive(Debug, PartialEq, Clone)]
pub struct Type(pub ResultType, pub ResultType);

impl Type {
    pub fn input_arity(&self) -> usize {
        self.0 .0.len()
    }

    pub fn output_arity(&self) -> usize {
        self.1 .0.len()
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Limits {
    Min(u32),
    Range(u32, u32),
}

impl Limits {
    pub fn min(&self) -> u32 {
        *match self {
            Limits::Min(min) => min,
            Limits::Range(min, _) => min,
        }
    }

    pub fn max(&self) -> Option<u32> {
        match self {
            Limits::Min(_) => None,
            Limits::Range(_, max) => Some(*max),
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct MemType(pub Limits);

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct TableType(pub RefType, pub Limits);

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Mutability {
    Const,
    Variable,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct GlobalType(pub ValType, pub Mutability);

#[derive(Debug, PartialEq, Clone)]
pub struct Global(pub GlobalType, pub Expr);

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BlockType {
    Empty,
    Val(ValType),
    TypeIndex(TypeIdx),
}

/// # MemArg
///
/// A memarg comprises two elements: an offset and an alignment.
///
/// The multiple memory proposal extends this with a memory index.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct MemArg(pub u32, pub u32);

impl MemArg {
    pub fn memidx(&self) -> usize {
        0
    }

    pub fn offset(&self) -> usize {
        self.1 as usize
    }

    pub fn align(&self) -> usize {
        self.0 as usize
    }
}

// TODO(maybe-optimize): split "Instr type" from "Instr payload". We can pack a lot more instrs
// into the processing stream this way; instrs fit into a single byte, giving us three bytes to
// index into a separate "instr payload" stream as necessary (16,777,215 values per stream; plus
// we can deduplicate repeated payloads.)
//
// Otherwise instrs take up 40 bytes (!!) which is kinda too many bytes?
/*pub enum InstrPayload {
    TypeInstrs(BlockType, Box<[Instr]>),
    TypeInstrs2(BlockType, Box<[Instr]>, Box<[Instr]>),
    BrTable(Box<[LabelIdx]>, LabelIdx),
    Select(Box<[ValType]>),
    DataOp(DataIdx),
    MemDataOp(DataIdx, MemIdx),
    ElemOp(ElemIdx),
    TableElemOp(ElemIdx, TableIdx),
    FuncOp(FuncIdx),
    GlobalOp(GlobalIdx),
    LabelOp(LabelIdx),
    LocalOp(LocalIdx),
    LoadOp(MemArg),
    MemOp(MemIdx),
    MemOp2(MemIdx, MemIdx),
    RefOp(RefType),
    TableOp(TableIdx),
    TableOp2(TableIdx, TableIdx),
    TypeTableOp(TypeIdx, TableIdx),
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
}
*/

#[derive(Debug, PartialEq, Clone)]
pub enum Instr {
    // Internal (non-wasm) instructions
    CallIntrinsic(usize),

    // Control Instructions
    Unreachable,
    Nop,
    Block(BlockType, Box<[Instr]>),
    Loop(BlockType, Box<[Instr]>),
    If(BlockType, Box<[Instr]>),
    IfElse(BlockType, Box<[Instr]>, Box<[Instr]>),
    Br(LabelIdx),
    BrIf(LabelIdx),
    BrTable(Box<[LabelIdx]>, LabelIdx),
    Return,
    Call(FuncIdx),
    CallIndirect(TypeIdx, TableIdx),

    // Reference Instructions
    RefNull(RefType),
    RefIsNull,
    RefFunc(FuncIdx),

    // Parametric Instructions
    Drop,
    SelectEmpty,
    Select(Box<[ValType]>),

    // Variable Instructions
    LocalGet(LocalIdx),
    LocalSet(LocalIdx),
    LocalTee(LocalIdx),
    GlobalGet(GlobalIdx),
    GlobalSet(GlobalIdx),

    // Table Instructions
    TableGet(TableIdx),
    TableSet(TableIdx),
    TableInit(ElemIdx, TableIdx),
    ElemDrop(ElemIdx),
    TableCopy(TableIdx, TableIdx),
    TableGrow(TableIdx),
    TableSize(TableIdx),
    TableFill(TableIdx),

    // Memory Instructions
    I32Load(MemArg),
    I64Load(MemArg),
    F32Load(MemArg),
    F64Load(MemArg),
    I32Load8S(MemArg),
    I32Load8U(MemArg),
    I32Load16S(MemArg),
    I32Load16U(MemArg),
    I64Load8S(MemArg),
    I64Load8U(MemArg),
    I64Load16S(MemArg),
    I64Load16U(MemArg),
    I64Load32S(MemArg),
    I64Load32U(MemArg),
    I32Store(MemArg),
    I64Store(MemArg),
    F32Store(MemArg),
    F64Store(MemArg),
    I32Store8(MemArg),
    I32Store16(MemArg),
    I64Store8(MemArg),
    I64Store16(MemArg),
    I64Store32(MemArg),
    MemorySize(MemIdx),
    MemoryGrow(MemIdx),
    MemoryInit(DataIdx, MemIdx),
    DataDrop(DataIdx),
    MemoryCopy(MemIdx, MemIdx),
    MemoryFill(MemIdx),

    // Numeric Instructions
    I32Const(i32),
    I64Const(i64),
    F32Const(f32),
    F64Const(f64),
    I32Eqz,
    I32Eq,
    I32Ne,
    I32LtS,
    I32LtU,
    I32GtS,
    I32GtU,
    I32LeS,
    I32LeU,
    I32GeS,
    I32GeU,
    I64Eqz,
    I64Eq,
    I64Ne,
    I64LtS,
    I64LtU,
    I64GtS,
    I64GtU,
    I64LeS,
    I64LeU,
    I64GeS,
    I64GeU,
    F32Eq,
    F32Ne,
    F32Lt,
    F32Gt,
    F32Le,
    F32Ge,
    F64Eq,
    F64Ne,
    F64Lt,
    F64Gt,
    F64Le,
    F64Ge,
    I32Clz,
    I32Ctz,
    I32Popcnt,
    I32Add,
    I32Sub,
    I32Mul,

    I32DivS,
    I32DivU,
    I32RemS,
    I32RemU,
    I32And,
    I32Ior,
    I32Xor,
    I32Shl,
    I32ShrS,
    I32ShrU,
    I32Rol,
    I32Ror,
    I64Clz,
    I64Ctz,
    I64Popcnt,
    I64Add,
    I64Sub,
    I64Mul,

    I64DivS,
    I64DivU,
    I64RemS,
    I64RemU,
    I64And,
    I64Ior,
    I64Xor,
    I64Shl,
    I64ShrS,
    I64ShrU,
    I64Rol,
    I64Ror,
    F32Abs,
    F32Neg,
    F32Ceil,
    F32Floor,
    F32Trunc,
    F32NearestInt,
    F32Sqrt,
    F32Add,
    F32Sub,
    F32Mul,
    F32Div,
    F32Min,
    F32Max,
    F32CopySign,
    F64Abs,
    F64Neg,
    F64Ceil,
    F64Floor,
    F64Trunc,
    F64NearestInt,
    F64Sqrt,
    F64Add,
    F64Sub,
    F64Mul,
    F64Div,
    F64Min,
    F64Max,
    F64CopySign,
    I32ConvertI64,
    I32SConvertF32,
    I32UConvertF32,
    I32SConvertF64,
    I32UConvertF64,
    I64SConvertI32,
    I64UConvertI32,
    I64SConvertF32,
    I64UConvertF32,
    I64SConvertF64,
    I64UConvertF64,
    F32SConvertI32,
    F32UConvertI32,
    F32SConvertI64,
    F32UConvertI64,
    F32ConvertF64,
    F64SConvertI32,
    F64UConvertI32,
    F64SConvertI64,
    F64UConvertI64,
    F64ConvertF32,

    I32SConvertSatF32,
    I32UConvertSatF32,
    I32SConvertSatF64,
    I32UConvertSatF64,
    I64SConvertSatF32,
    I64UConvertSatF32,
    I64SConvertSatF64,
    I64UConvertSatF64,

    I32ReinterpretF32,
    I64ReinterpretF64,
    F32ReinterpretI32,
    F64ReinterpretI64,
    I32SExtendI8,
    I32SExtendI16,
    I64SExtendI8,
    I64SExtendI16,
    I64SExtendI32,
    // Vector Instructions
}

#[derive(Debug, PartialEq, Clone)]
pub struct Expr(pub Vec<Instr>);

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(transparent)]
pub struct TypeIdx(pub u32);

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct FuncIdx(pub u32);

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(transparent)]
pub struct CodeIdx(pub u32);

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct TableIdx(pub u32);

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct MemIdx(pub u32);

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct GlobalIdx(pub u32);

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct ElemIdx(pub u32);

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct DataIdx(pub u32);

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct LocalIdx(pub u32);

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct LabelIdx(pub u32);

#[derive(Debug, PartialEq, Clone)]
pub struct Import {
    pub(crate) r#mod: Name,
    pub(crate) nm: Name,
    pub(crate) desc: ImportDesc,
}

impl Import {
    pub fn new(module: Name, name: Name, desc: ImportDesc) -> Self {
        Self {
            r#mod: module,
            nm: name,
            desc,
        }
    }

    pub fn module(&self) -> &str {
        &self.r#mod.0
    }

    pub fn name(&self) -> &str {
        &self.nm.0
    }

    pub fn desc(&self) -> &ImportDesc {
        &self.desc
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ImportDesc {
    Func(TypeIdx),
    Table(TableType),
    Mem(MemType),
    Global(GlobalType),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Export {
    pub(crate) nm: Name,
    pub(crate) desc: ExportDesc,
}

impl Export {
    pub fn new(name: String, desc: ExportDesc) -> Self {
        Self {
            nm: Name(name),
            desc,
        }
    }

    pub fn name(&self) -> &str {
        &self.nm.0
    }

    pub fn desc(&self) -> &ExportDesc {
        &self.desc
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ExportDesc {
    Func(FuncIdx),
    Table(TableIdx),
    Mem(MemIdx),
    Global(GlobalIdx),
}

#[derive(Debug, PartialEq, Clone)]
pub enum ElemMode {
    Passive,
    Active { table_idx: TableIdx, offset: Expr },
    Declarative,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Elem {
    pub mode: ElemMode,
    pub kind: RefType,
    pub exprs: Box<[Expr]>,
    /// The original flags the Elem was encoded with, so we can round-trip an elem.
    pub flags: u8,
}

impl Elem {
    pub fn len(&self) -> usize {
        self.exprs.len()
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Local(pub u32, pub ValType);

impl From<Type> for Vec<Local> {
    fn from(value: Type) -> Self {
        value.0 .0.iter().map(|xs| Local(0, *xs)).collect()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Func {
    pub locals: Box<[Local]>,
    pub expr: Expr,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Code(pub Func);

#[derive(Debug, PartialEq, Clone)]
pub enum Data {
    Active(ByteVec, MemIdx, Expr),
    Passive(ByteVec),
}

#[derive(Debug, PartialEq, Clone)]
pub enum SectionType {
    Custom(Box<[u8]>),
    Type(Box<[Type]>),
    Import(Box<[Import]>),
    Function(Box<[TypeIdx]>),
    Table(Box<[TableType]>),
    Memory(Box<[MemType]>),
    Global(Box<[Global]>),
    Export(Box<[Export]>),
    Start(FuncIdx),
    Element(Box<[Elem]>),
    Code(Box<[Code]>),
    Data(Box<[Data]>),
    DataCount(u32),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Section<T: Debug + PartialEq + Clone> {
    pub index: usize,
    pub inner: T,
}

#[derive(Debug, PartialEq, Clone, Default)]
pub struct Module {
    pub(crate) custom_sections: Vec<Section<Box<[u8]>>>,
    pub(crate) type_section: Option<Section<Box<[Type]>>>,
    pub(crate) import_section: Option<Section<Box<[Import]>>>,
    pub(crate) function_section: Option<Section<Box<[TypeIdx]>>>,
    pub(crate) table_section: Option<Section<Box<[TableType]>>>,
    pub(crate) memory_section: Option<Section<Box<[MemType]>>>,
    pub(crate) global_section: Option<Section<Box<[Global]>>>,
    pub(crate) export_section: Option<Section<Box<[Export]>>>,
    pub(crate) start_section: Option<Section<FuncIdx>>,
    pub(crate) element_section: Option<Section<Box<[Elem]>>>,
    pub(crate) code_section: Option<Section<Box<[Code]>>>,
    pub(crate) data_section: Option<Section<Box<[Data]>>>,
    pub(crate) datacount_section: Option<Section<u32>>,
}

/// A struct used by [`Module::into_inner`] to expose all module contents
/// to consumers.
pub struct ModuleIntoInner {
    pub custom_sections: Vec<Section<Box<[u8]>>>,
    pub type_section: Option<Section<Box<[Type]>>>,
    pub import_section: Option<Section<Box<[Import]>>>,
    pub function_section: Option<Section<Box<[TypeIdx]>>>,
    pub table_section: Option<Section<Box<[TableType]>>>,
    pub memory_section: Option<Section<Box<[MemType]>>>,
    pub global_section: Option<Section<Box<[Global]>>>,
    pub export_section: Option<Section<Box<[Export]>>>,
    pub start_section: Option<Section<FuncIdx>>,
    pub element_section: Option<Section<Box<[Elem]>>>,
    pub code_section: Option<Section<Box<[Code]>>>,
    pub data_section: Option<Section<Box<[Data]>>>,
    pub datacount_section: Option<Section<u32>>,
}

#[derive(Default, Debug)]
pub struct ModuleBuilder {
    inner: Module,
    index: usize,
}

impl ModuleBuilder {
    #[inline]
    pub fn new() -> Self {
        Default::default()
    }

    pub fn custom_section(mut self, v: Box<[u8]>) -> Self {
        self.inner.custom_sections.push(Section {
            inner: v,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub fn type_section(mut self, xs: Box<[Type]>) -> Self {
        self.inner.type_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub fn import_section(mut self, xs: Box<[Import]>) -> Self {
        self.inner.import_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub fn function_section(mut self, xs: Box<[TypeIdx]>) -> Self {
        self.inner.function_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub fn table_section(mut self, xs: Box<[TableType]>) -> Self {
        self.inner.table_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub fn memory_section(mut self, xs: Box<[MemType]>) -> Self {
        self.inner.memory_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub fn global_section(mut self, xs: Box<[Global]>) -> Self {
        self.inner.global_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub fn export_section(mut self, xs: Box<[Export]>) -> Self {
        self.inner.export_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub fn start_section(mut self, xs: FuncIdx) -> Self {
        self.inner.start_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub fn element_section(mut self, xs: Box<[Elem]>) -> Self {
        self.inner.element_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub fn code_section(mut self, xs: Box<[Code]>) -> Self {
        self.inner.code_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub fn data_section(mut self, xs: Box<[Data]>) -> Self {
        self.inner.data_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub fn datacount_section(mut self, xs: u32) -> Self {
        self.inner.datacount_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }

    pub fn build(self) -> Module {
        self.inner
    }
}

impl Module {
    pub fn into_inner(self) -> ModuleIntoInner {
        let Self {
            custom_sections,
            type_section,
            import_section,
            function_section,
            table_section,
            memory_section,
            global_section,
            export_section,
            start_section,
            element_section,
            code_section,
            data_section,
            datacount_section,
        } = self;
        ModuleIntoInner {
            custom_sections,
            type_section,
            import_section,
            function_section,
            table_section,
            memory_section,
            global_section,
            export_section,
            start_section,
            element_section,
            code_section,
            data_section,
            datacount_section,
        }
    }

    pub fn custom_sections(&self) -> impl Iterator<Item = &[u8]> {
        self.custom_sections.iter().map(|xs| &*xs.inner)
    }

    pub fn type_section(&self) -> Option<&[Type]> {
        self.type_section.as_ref().map(|xs| &*xs.inner)
    }

    pub fn import_section(&self) -> Option<&[Import]> {
        self.import_section.as_ref().map(|xs| &*xs.inner)
    }

    pub fn function_section(&self) -> Option<&[TypeIdx]> {
        self.function_section.as_ref().map(|xs| &*xs.inner)
    }

    pub fn table_section(&self) -> Option<&[TableType]> {
        self.table_section.as_ref().map(|xs| &*xs.inner)
    }

    pub fn memory_section(&self) -> Option<&[MemType]> {
        self.memory_section.as_ref().map(|xs| &*xs.inner)
    }

    pub fn global_section(&self) -> Option<&[Global]> {
        self.global_section.as_ref().map(|xs| &*xs.inner)
    }

    pub fn export_section(&self) -> Option<&[Export]> {
        self.export_section.as_ref().map(|xs| &*xs.inner)
    }

    pub fn start_section(&self) -> Option<FuncIdx> {
        self.start_section.as_ref().map(|xs| xs.inner)
    }

    pub fn element_section(&self) -> Option<&[Elem]> {
        self.element_section.as_ref().map(|xs| &*xs.inner)
    }

    pub fn code_section(&self) -> Option<&[Code]> {
        self.code_section.as_ref().map(|xs| &*xs.inner)
    }

    pub fn data_section(&self) -> Option<&[Data]> {
        self.data_section.as_ref().map(|xs| &*xs.inner)
    }

    pub fn datacount_section(&self) -> Option<u32> {
        self.datacount_section.as_ref().map(|xs| xs.inner)
    }
}

pub trait IR {
    type Error: Clone + Error + 'static;

    type BlockType;
    type MemType;
    type ByteVec;
    type Code;
    type CodeIdx;
    type Data;
    type DataIdx;
    type ElemMode;
    type Elem;
    type ElemIdx;
    type Export;
    type ExportDesc;
    type Expr;
    type Func;
    type FuncIdx;
    type Global;
    type GlobalIdx;
    type GlobalType;
    type Import;
    type ImportDesc;
    type Instr;
    type LabelIdx;
    type Limits;
    type Local;
    type LocalIdx;
    type MemArg;
    type MemIdx;
    type Module;
    type Name;
    type NumType;
    type RefType;
    type ResultType;
    type Section;
    type TableIdx;
    type TableType;
    type Type;
    type TypeIdx;
    type ValType;
    type VecType;

    fn make_instr_select(
        &mut self,
        types: Box<[Self::ValType]>,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error>;

    fn make_instr_table(
        &mut self,
        items: Box<[u32]>,
        alternate: u32,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error>;

    fn make_instr_unary64(
        &mut self,
        code: u8,
        arg0: u64,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error>;

    fn make_instr_binary(
        &mut self,
        code: u8,
        arg0: u32,
        arg1: u32,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error>;

    fn make_instr_unary(
        &mut self,
        code: u8,
        arg0: u32,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error>;

    fn make_instr_nullary(
        &mut self,
        code: u8,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error>;

    fn make_instr_block(
        &mut self,
        block_kind: u8,
        block_type: Self::BlockType,
        expr: Self::Expr,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error>;

    fn make_instr_block_ifelse(
        &mut self,
        block_type: Self::BlockType,
        consequent: Self::Expr,
        alternate: Option<Self::Expr>,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error>;

    fn make_code(&mut self, item: Self::Func) -> Result<Self::Code, Self::Error>;

    fn make_data_active(
        &mut self,
        bytes: Box<[u8]>,
        mem_idx: Self::MemIdx,
        expr: Self::Expr,
    ) -> Result<Self::Data, Self::Error>;
    fn make_data_passive(&mut self, bytes: Box<[u8]>) -> Result<Self::Data, Self::Error>;

    fn make_elem_from_indices(
        &mut self,
        kind: Option<u32>,
        mode: Self::ElemMode,
        idxs: Box<[u32]>,
        flags: u8,
    ) -> Result<Self::Elem, Self::Error>;
    fn make_elem_from_exprs(
        &mut self,
        kind: Option<Self::RefType>,
        mode: Self::ElemMode,
        exprs: Box<[Self::Expr]>,
        flags: u8,
    ) -> Result<Self::Elem, Self::Error>;

    fn make_elem_mode_passive(&mut self) -> Result<Self::ElemMode, Self::Error>;
    fn make_elem_mode_declarative(&mut self) -> Result<Self::ElemMode, Self::Error>;
    fn make_elem_mode_active(
        &mut self,
        table_idx: Self::TableIdx,
        expr: Self::Expr,
    ) -> Result<Self::ElemMode, Self::Error>;

    fn make_limits(&mut self, lower: u32, upper: Option<u32>) -> Result<Self::Limits, Self::Error>;

    fn make_export(
        &mut self,
        name: Self::Name,
        desc: Self::ExportDesc,
    ) -> Result<Self::Export, Self::Error>;

    fn make_expr(&mut self, instrs: Vec<Self::Instr>) -> Result<Self::Expr, Self::Error>;

    fn start_func(&mut self) {}

    fn make_func(
        &mut self,
        locals: Box<[Self::Local]>,
        expr: Self::Expr,
    ) -> Result<Self::Func, Self::Error>;

    fn make_global(
        &mut self,
        global_type: Self::GlobalType,
        expr: Self::Expr,
    ) -> Result<Self::Global, Self::Error>;

    fn make_local(
        &mut self,
        count: u32,
        val_type: Self::ValType,
    ) -> Result<Self::Local, Self::Error>;

    fn make_name(&mut self, data: Box<[u8]>) -> Result<Self::Name, Self::Error>;
    fn make_custom_section(&mut self, data: Box<[u8]>) -> Result<Self::Section, Self::Error>;
    fn make_type_section(&mut self, data: Box<[Self::Type]>) -> Result<Self::Section, Self::Error>;
    fn make_import_section(
        &mut self,
        data: Box<[Self::Import]>,
    ) -> Result<Self::Section, Self::Error>;
    fn make_function_section(
        &mut self,
        data: Box<[Self::TypeIdx]>,
    ) -> Result<Self::Section, Self::Error>;
    fn make_table_section(
        &mut self,
        data: Box<[Self::TableType]>,
    ) -> Result<Self::Section, Self::Error>;
    fn make_memory_section(
        &mut self,
        data: Box<[Self::MemType]>,
    ) -> Result<Self::Section, Self::Error>;
    fn make_global_section(
        &mut self,
        data: Box<[Self::Global]>,
    ) -> Result<Self::Section, Self::Error>;
    fn make_export_section(
        &mut self,
        data: Box<[Self::Export]>,
    ) -> Result<Self::Section, Self::Error>;

    fn make_start_section(&mut self, data: Self::FuncIdx) -> Result<Self::Section, Self::Error>;

    fn make_element_section(
        &mut self,
        data: Box<[Self::Elem]>,
    ) -> Result<Self::Section, Self::Error>;

    fn make_code_section(&mut self, data: Box<[Self::Code]>) -> Result<Self::Section, Self::Error>;
    fn make_data_section(&mut self, data: Box<[Self::Data]>) -> Result<Self::Section, Self::Error>;

    fn make_datacount_section(&mut self, data: u32) -> Result<Self::Section, Self::Error>;

    fn make_block_type_empty(&mut self) -> Result<Self::BlockType, Self::Error>;
    fn make_block_type_val_type(
        &mut self,
        vt: Self::ValType,
    ) -> Result<Self::BlockType, Self::Error>;
    fn make_block_type_type_index(
        &mut self,
        ti: Self::TypeIdx,
    ) -> Result<Self::BlockType, Self::Error>;

    fn make_val_type(&mut self, data: u8) -> Result<Self::ValType, Self::Error>;
    fn make_ref_type(&mut self, data: u8) -> Result<Self::RefType, Self::Error>;
    fn make_global_type(
        &mut self,
        valtype: Self::ValType,
        is_mutable: bool,
    ) -> Result<Self::GlobalType, Self::Error>;
    fn make_table_type(
        &mut self,
        reftype_candidate: u8,
        limits: Self::Limits,
    ) -> Result<Self::TableType, Self::Error>;
    fn make_mem_type(&mut self, limits: Self::Limits) -> Result<Self::MemType, Self::Error>;

    fn make_result_type(&mut self, data: &[u8]) -> Result<Self::ResultType, Self::Error>;
    fn make_type_index(&mut self, candidate: u32) -> Result<Self::TypeIdx, Self::Error>;
    fn make_table_index(&mut self, candidate: u32) -> Result<Self::TableIdx, Self::Error>;
    fn make_mem_index(&mut self, candidate: u32) -> Result<Self::MemIdx, Self::Error>;
    fn make_global_index(&mut self, candidate: u32) -> Result<Self::GlobalIdx, Self::Error>;
    fn make_func_index(&mut self, candidate: u32) -> Result<Self::FuncIdx, Self::Error>;
    fn make_local_index(&mut self, candidate: u32) -> Result<Self::LocalIdx, Self::Error>;
    fn make_func_type(
        &mut self,
        params: Option<Self::ResultType>,
        returns: Option<Self::ResultType>,
    ) -> Result<Self::Type, Self::Error>;

    fn make_import_desc_func(
        &mut self,
        type_idx: Self::TypeIdx,
    ) -> Result<Self::ImportDesc, Self::Error>;
    fn make_import_desc_global(
        &mut self,
        global_type: Self::GlobalType,
    ) -> Result<Self::ImportDesc, Self::Error>;
    fn make_import_desc_table(
        &mut self,
        global_type: Self::TableType,
    ) -> Result<Self::ImportDesc, Self::Error>;
    fn make_import_desc_memtype(
        &mut self,
        global_type: Self::Limits,
    ) -> Result<Self::ImportDesc, Self::Error>;

    fn make_export_desc_func(
        &mut self,
        func_idx: Self::FuncIdx,
    ) -> Result<Self::ExportDesc, Self::Error>;

    fn make_export_desc_global(
        &mut self,
        global_idx: Self::GlobalIdx,
    ) -> Result<Self::ExportDesc, Self::Error>;

    fn make_export_desc_memtype(
        &mut self,
        mem_idx: Self::MemIdx,
    ) -> Result<Self::ExportDesc, Self::Error>;

    fn make_export_desc_table(
        &mut self,
        table_idx: Self::TableIdx,
    ) -> Result<Self::ExportDesc, Self::Error>;

    fn make_import(
        &mut self,
        modname: Self::Name,
        name: Self::Name,
        desc: Self::ImportDesc,
    ) -> Result<Self::Import, Self::Error>;
    fn make_module(&mut self, sections: Vec<Self::Section>) -> Result<Self::Module, Self::Error>;
}

#[derive(Default, Clone, Debug)]
pub struct EmptyIRGenerator;

impl EmptyIRGenerator {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Clone, Debug)]
pub enum Never {}
impl Error for Never {}
impl std::fmt::Display for Never {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("never!")
    }
}

impl IR for EmptyIRGenerator {
    type Error = Never;

    type BlockType = ();
    type MemType = ();
    type ByteVec = ();
    type Code = ();
    type CodeIdx = ();
    type Data = ();
    type DataIdx = ();
    type Elem = ();
    type ElemMode = ();
    type ElemIdx = ();
    type Export = ();
    type ExportDesc = ();
    type Expr = ();
    type Func = ();
    type FuncIdx = ();
    type Global = ();
    type GlobalIdx = ();
    type GlobalType = ();
    type Import = ();
    type ImportDesc = ();
    type Instr = ();
    type LabelIdx = ();
    type Limits = ();
    type Local = ();
    type LocalIdx = ();
    type MemArg = ();
    type MemIdx = ();
    type Module = ();
    type Name = ();
    type NumType = ();
    type RefType = ();
    type ResultType = ();
    type Section = ();
    type TableIdx = ();
    type TableType = ();
    type Type = ();
    type TypeIdx = ();
    type ValType = ();
    type VecType = ();

    fn make_instr_select(
        &mut self,
        types: Box<[Self::ValType]>,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn make_instr_table(
        &mut self,
        items: Box<[u32]>,
        alternate: u32,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn make_instr_unary64(
        &mut self,
        code: u8,
        arg0: u64,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn make_instr_binary(
        &mut self,
        code: u8,
        arg0: u32,
        arg1: u32,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn make_instr_unary(
        &mut self,
        code: u8,
        arg0: u32,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn make_instr_nullary(
        &mut self,
        code: u8,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn make_instr_block(
        &mut self,
        block_kind: u8,
        block_type: Self::BlockType,
        expr: Self::Expr,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn make_instr_block_ifelse(
        &mut self,
        block_type: Self::BlockType,
        consequent: Self::Expr,
        alternate: Option<Self::Expr>,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn make_code(&mut self, item: Self::Func) -> Result<Self::Code, Self::Error> {
        Ok(())
    }

    fn make_data_active(
        &mut self,
        bytes: Box<[u8]>,
        mem_idx: Self::MemIdx,
        expr: Self::Expr,
    ) -> Result<Self::Data, Self::Error> {
        Ok(())
    }
    fn make_data_passive(&mut self, bytes: Box<[u8]>) -> Result<Self::Data, Self::Error> {
        Ok(())
    }

    fn make_limits(&mut self, lower: u32, upper: Option<u32>) -> Result<Self::Limits, Self::Error> {
        Ok(())
    }

    fn make_export(
        &mut self,
        name: Self::Name,
        desc: Self::ExportDesc,
    ) -> Result<Self::Export, Self::Error> {
        Ok(())
    }

    fn make_expr(&mut self, instrs: Vec<Self::Instr>) -> Result<Self::Expr, Self::Error> {
        Ok(())
    }

    fn make_func(
        &mut self,
        locals: Box<[Self::Local]>,
        expr: Self::Expr,
    ) -> Result<Self::Func, Self::Error> {
        Ok(())
    }

    fn make_global(
        &mut self,
        global_type: Self::GlobalType,
        expr: Self::Expr,
    ) -> Result<Self::Global, Self::Error> {
        Ok(())
    }

    fn make_local(
        &mut self,
        count: u32,
        val_type: Self::ValType,
    ) -> Result<Self::Local, Self::Error> {
        Ok(())
    }

    fn make_name(&mut self, data: Box<[u8]>) -> Result<Self::Name, Self::Error> {
        Ok(())
    }
    fn make_custom_section(&mut self, data: Box<[u8]>) -> Result<Self::Section, Self::Error> {
        Ok(())
    }
    fn make_type_section(&mut self, data: Box<[Self::Type]>) -> Result<Self::Section, Self::Error> {
        Ok(())
    }
    fn make_import_section(
        &mut self,
        data: Box<[Self::Import]>,
    ) -> Result<Self::Section, Self::Error> {
        Ok(())
    }
    fn make_function_section(
        &mut self,
        data: Box<[Self::TypeIdx]>,
    ) -> Result<Self::Section, Self::Error> {
        Ok(())
    }
    fn make_table_section(
        &mut self,
        data: Box<[Self::TableType]>,
    ) -> Result<Self::Section, Self::Error> {
        Ok(())
    }
    fn make_memory_section(
        &mut self,
        data: Box<[Self::MemType]>,
    ) -> Result<Self::Section, Self::Error> {
        Ok(())
    }
    fn make_global_section(
        &mut self,
        data: Box<[Self::Global]>,
    ) -> Result<Self::Section, Self::Error> {
        Ok(())
    }
    fn make_export_section(
        &mut self,
        data: Box<[Self::Export]>,
    ) -> Result<Self::Section, Self::Error> {
        Ok(())
    }

    fn make_start_section(&mut self, data: Self::FuncIdx) -> Result<Self::Section, Self::Error> {
        Ok(())
    }

    fn make_element_section(
        &mut self,
        data: Box<[Self::Elem]>,
    ) -> Result<Self::Section, Self::Error> {
        Ok(())
    }

    fn make_code_section(&mut self, data: Box<[Self::Code]>) -> Result<Self::Section, Self::Error> {
        Ok(())
    }
    fn make_data_section(&mut self, data: Box<[Self::Data]>) -> Result<Self::Section, Self::Error> {
        Ok(())
    }

    fn make_datacount_section(&mut self, data: u32) -> Result<Self::Section, Self::Error> {
        Ok(())
    }

    fn make_block_type_empty(&mut self) -> Result<Self::BlockType, Self::Error> {
        Ok(())
    }
    fn make_block_type_val_type(
        &mut self,
        vt: Self::ValType,
    ) -> Result<Self::BlockType, Self::Error> {
        Ok(())
    }
    fn make_block_type_type_index(
        &mut self,
        ti: Self::TypeIdx,
    ) -> Result<Self::BlockType, Self::Error> {
        Ok(())
    }

    fn make_val_type(&mut self, data: u8) -> Result<Self::ValType, Self::Error> {
        Ok(())
    }
    fn make_global_type(
        &mut self,
        valtype: Self::ValType,
        is_mutable: bool,
    ) -> Result<Self::GlobalType, Self::Error> {
        Ok(())
    }
    fn make_table_type(
        &mut self,
        reftype_candidate: u8,
        limits: Self::Limits,
    ) -> Result<Self::TableType, Self::Error> {
        Ok(())
    }
    fn make_mem_type(&mut self, limits: Self::Limits) -> Result<Self::MemType, Self::Error> {
        Ok(())
    }

    fn make_result_type(&mut self, data: &[u8]) -> Result<Self::ResultType, Self::Error> {
        Ok(())
    }
    fn make_type_index(&mut self, candidate: u32) -> Result<Self::TypeIdx, Self::Error> {
        Ok(())
    }
    fn make_table_index(&mut self, candidate: u32) -> Result<Self::TableIdx, Self::Error> {
        Ok(())
    }
    fn make_mem_index(&mut self, candidate: u32) -> Result<Self::MemIdx, Self::Error> {
        Ok(())
    }
    fn make_global_index(&mut self, candidate: u32) -> Result<Self::GlobalIdx, Self::Error> {
        Ok(())
    }
    fn make_func_index(&mut self, candidate: u32) -> Result<Self::FuncIdx, Self::Error> {
        Ok(())
    }
    fn make_local_index(&mut self, candidate: u32) -> Result<Self::FuncIdx, Self::Error> {
        Ok(())
    }
    fn make_func_type(
        &mut self,
        params: Option<Self::ResultType>,
        returns: Option<Self::ResultType>,
    ) -> Result<Self::Type, Self::Error> {
        Ok(())
    }

    fn make_import_desc_func(
        &mut self,
        type_idx: Self::TypeIdx,
    ) -> Result<Self::ImportDesc, Self::Error> {
        Ok(())
    }
    fn make_import_desc_global(
        &mut self,
        global_type: Self::GlobalType,
    ) -> Result<Self::ImportDesc, Self::Error> {
        Ok(())
    }
    fn make_import_desc_table(
        &mut self,
        global_type: Self::TableType,
    ) -> Result<Self::ImportDesc, Self::Error> {
        Ok(())
    }
    fn make_import_desc_memtype(
        &mut self,
        global_type: Self::Limits,
    ) -> Result<Self::ImportDesc, Self::Error> {
        Ok(())
    }

    fn make_export_desc_func(
        &mut self,
        func_idx: Self::FuncIdx,
    ) -> Result<Self::ExportDesc, Self::Error> {
        Ok(())
    }

    fn make_export_desc_global(
        &mut self,
        global_idx: Self::GlobalIdx,
    ) -> Result<Self::ExportDesc, Self::Error> {
        Ok(())
    }

    fn make_export_desc_memtype(
        &mut self,
        mem_idx: Self::MemIdx,
    ) -> Result<Self::ExportDesc, Self::Error> {
        Ok(())
    }

    fn make_export_desc_table(
        &mut self,
        table_idx: Self::TableIdx,
    ) -> Result<Self::ExportDesc, Self::Error> {
        Ok(())
    }

    fn make_import(
        &mut self,
        modname: Self::Name,
        name: Self::Name,
        desc: Self::ImportDesc,
    ) -> Result<Self::Import, Self::Error> {
        Ok(())
    }
    fn make_module(&mut self, sections: Vec<Self::Section>) -> Result<Self::Module, Self::Error> {
        Ok(())
    }

    fn make_elem_from_indices(
        &mut self,
        _kind: Option<u32>,
        _mode: Self::ElemMode,
        _idxs: Box<[u32]>,
        _flags: u8,
    ) -> Result<Self::Elem, Self::Error> {
        Ok(())
    }

    fn make_elem_from_exprs(
        &mut self,
        _kind: Option<Self::RefType>,
        _mode: Self::ElemMode,
        _exprs: Box<[Self::Expr]>,
        _flags: u8,
    ) -> Result<Self::Elem, Self::Error> {
        Ok(())
    }

    fn make_elem_mode_passive(&mut self) -> Result<Self::ElemMode, Self::Error> {
        Ok(())
    }

    fn make_elem_mode_declarative(&mut self) -> Result<Self::ElemMode, Self::Error> {
        Ok(())
    }

    fn make_elem_mode_active(
        &mut self,
        table_idx: Self::TableIdx,
        expr: Self::Expr,
    ) -> Result<Self::ElemMode, Self::Error> {
        Ok(())
    }

    fn make_ref_type(&mut self, data: u8) -> Result<Self::RefType, Self::Error> {
        Ok(())
    }
}

#[derive(Clone, Debug, Error)]
pub enum DefaultIRGeneratorError {
    #[error("Invalid name: {0}")]
    InvalidName(#[from] std::str::Utf8Error),

    #[error("Invalid instruction (got {0:X}H)")]
    InvalidInstruction(u8),

    #[error("Invalid type (got {0:X}H)")]
    InvalidType(u8),

    #[error("Invalid type index (got {0}; max is {1})")]
    InvalidTypeIndex(u32, u32),

    #[error("Invalid global index (got {0}; max is {1})")]
    InvalidGlobalIndex(u32, u32),

    #[error("Invalid table index (got {0}; max is {1})")]
    InvalidTableIndex(u32, u32),

    #[error("Invalid local index (got {0}; max is {1})")]
    InvalidLocalIndex(u32, u32),

    #[error("Invalid func index (got {0}; max is {1})")]
    InvalidFuncIndex(u32, u32),

    #[error("Invalid memory index (got {0}; max is {1})")]
    InvalidMemIndex(u32, u32),

    #[error("Types out of order (got section type {0} after type {1})")]
    InvalidSectionOrder(u32, u32),

    #[error("Datacount section value did not match data element count (expected {0}, got {1})")]
    DatacountMismatch(u32, u32),

    #[error("Invalid reference type {0}")]
    InvalidRefType(u8),
}

#[derive(Default, Clone, Debug)]
pub struct DefaultIRGenerator {
    max_valid_type_index: u32,
    max_valid_func_index: u32,
    max_valid_table_index: u32,
    max_valid_global_index: u32,
    max_valid_element_index: u32,
    max_valid_data_index: Option<u32>,
    local_function_count: u32,
    max_valid_mem_index: u32,
    last_section_discrim: u32,

    next_func_idx: u32,

    current_locals: Vec<ValType>,
    blocks: Vec<BlockType>,
    stack: Vec<ValType>,
}

impl DefaultIRGenerator {
    pub fn new() -> Self {
        Self::default()
    }
}

impl IR for DefaultIRGenerator {
    type Error = DefaultIRGeneratorError;

    type BlockType = BlockType;
    type MemType = MemType;
    type ByteVec = ByteVec;
    type Code = Code;
    type CodeIdx = CodeIdx;
    type Data = Data;
    type DataIdx = DataIdx;
    type Elem = Elem;
    type ElemMode = ElemMode;
    type ElemIdx = ElemIdx;
    type Export = Export;
    type ExportDesc = ExportDesc;
    type Expr = Expr;
    type Func = Func;
    type FuncIdx = FuncIdx;
    type Global = Global;
    type GlobalIdx = GlobalIdx;
    type GlobalType = GlobalType;
    type Import = Import;
    type ImportDesc = ImportDesc;
    type Instr = Instr;
    type LabelIdx = LabelIdx;
    type Limits = Limits;
    type Local = Local;
    type LocalIdx = LocalIdx;
    type MemArg = MemArg;
    type MemIdx = MemIdx;
    type Module = Module;
    type Name = Name;
    type NumType = NumType;
    type RefType = RefType;
    type ResultType = ResultType;
    type Section = SectionType;
    type TableIdx = TableIdx;
    type TableType = TableType;
    type Type = Type;
    type TypeIdx = TypeIdx;
    type ValType = ValType;
    type VecType = VecType;

    fn make_name(&mut self, data: Box<[u8]>) -> Result<Self::Name, Self::Error> {
        let string = std::str::from_utf8(&data)?;
        Ok(Name(string.to_string()))
    }

    #[inline]
    fn make_val_type(&mut self, item: u8) -> Result<Self::ValType, Self::Error> {
        Ok(match item {
            0x6f => ValType::RefType(RefType::ExternRef),
            0x70 => ValType::RefType(RefType::FuncRef),
            0x7b => ValType::VecType(VecType::V128),
            0x7c => ValType::NumType(NumType::F64),
            0x7d => ValType::NumType(NumType::F32),
            0x7e => ValType::NumType(NumType::I64),
            0x7f => ValType::NumType(NumType::I32),
            byte => return Err(DefaultIRGeneratorError::InvalidType(byte)),
        })
    }

    fn make_block_type_empty(&mut self) -> Result<Self::BlockType, Self::Error> {
        Ok(BlockType::Empty)
    }

    fn make_block_type_val_type(
        &mut self,
        vt: Self::ValType,
    ) -> Result<Self::BlockType, Self::Error> {
        Ok(BlockType::Val(vt))
    }

    fn make_block_type_type_index(
        &mut self,
        ti: Self::TypeIdx,
    ) -> Result<Self::BlockType, Self::Error> {
        Ok(BlockType::TypeIndex(ti))
    }

    fn make_global_type(
        &mut self,
        valtype: Self::ValType,
        is_mutable: bool,
    ) -> Result<Self::GlobalType, Self::Error> {
        Ok(GlobalType(
            valtype,
            if is_mutable {
                Mutability::Variable
            } else {
                Mutability::Const
            },
        ))
    }

    fn make_table_type(
        &mut self,
        reftype_candidate: u8,
        limits: Self::Limits,
    ) -> Result<Self::TableType, Self::Error> {
        if let ValType::RefType(rt) = self.make_val_type(reftype_candidate)? {
            Ok(TableType(rt, limits))
        } else {
            Err(DefaultIRGeneratorError::InvalidRefType(reftype_candidate))
        }
    }

    fn make_mem_type(&mut self, limits: Self::Limits) -> Result<Self::MemType, Self::Error> {
        Ok(MemType(limits))
    }

    fn make_result_type(&mut self, data: &[u8]) -> Result<Self::ResultType, Self::Error> {
        let mut types = Vec::with_capacity(data.len());
        for item in data {
            types.push(self.make_val_type(*item)?);
        }
        Ok(ResultType(types.into()))
    }

    fn make_custom_section(&mut self, data: Box<[u8]>) -> Result<Self::Section, Self::Error> {
        Ok(SectionType::Custom(data))
    }

    fn make_type_section(&mut self, data: Box<[Type]>) -> Result<Self::Section, Self::Error> {
        if self.last_section_discrim > 0 {
            return Err(DefaultIRGeneratorError::InvalidSectionOrder(
                1,
                self.last_section_discrim,
            ));
        }

        self.max_valid_type_index = data.len() as u32;
        self.last_section_discrim = 1;
        Ok(SectionType::Type(data))
    }

    fn make_import_section(
        &mut self,
        data: Box<[Self::Import]>,
    ) -> Result<Self::Section, Self::Error> {
        if self.last_section_discrim > 1 {
            return Err(DefaultIRGeneratorError::InvalidSectionOrder(
                2,
                self.last_section_discrim,
            ));
        }
        self.last_section_discrim = 2;
        Ok(SectionType::Import(data))
    }

    fn make_function_section(
        &mut self,
        data: Box<[Self::TypeIdx]>,
    ) -> Result<Self::Section, Self::Error> {
        if self.last_section_discrim > 2 {
            return Err(DefaultIRGeneratorError::InvalidSectionOrder(
                3,
                self.last_section_discrim,
            ));
        }
        self.local_function_count = data.len() as u32;
        self.max_valid_func_index += self.local_function_count;
        self.last_section_discrim = 3;
        Ok(SectionType::Function(data))
    }

    fn make_table_section(
        &mut self,
        data: Box<[Self::TableType]>,
    ) -> Result<Self::Section, Self::Error> {
        if self.last_section_discrim > 3 {
            return Err(DefaultIRGeneratorError::InvalidSectionOrder(
                4,
                self.last_section_discrim,
            ));
        }
        self.max_valid_table_index += data.len() as u32;
        self.last_section_discrim = 4;
        Ok(SectionType::Table(data))
    }

    fn make_memory_section(
        &mut self,
        data: Box<[Self::MemType]>,
    ) -> Result<Self::Section, Self::Error> {
        if self.last_section_discrim > 4 {
            return Err(DefaultIRGeneratorError::InvalidSectionOrder(
                5,
                self.last_section_discrim,
            ));
        }
        self.max_valid_mem_index += data.len() as u32;
        self.last_section_discrim = 5;
        Ok(SectionType::Memory(data))
    }

    fn make_global_section(
        &mut self,
        data: Box<[Self::Global]>,
    ) -> Result<Self::Section, Self::Error> {
        if self.last_section_discrim > 5 {
            return Err(DefaultIRGeneratorError::InvalidSectionOrder(
                6,
                self.last_section_discrim,
            ));
        }
        self.max_valid_global_index += data.len() as u32;
        self.last_section_discrim = 6;
        Ok(SectionType::Global(data))
    }

    fn make_export_section(
        &mut self,
        data: Box<[Self::Export]>,
    ) -> Result<Self::Section, Self::Error> {
        if self.last_section_discrim > 6 {
            return Err(DefaultIRGeneratorError::InvalidSectionOrder(
                7,
                self.last_section_discrim,
            ));
        }
        self.last_section_discrim = 7;
        Ok(SectionType::Export(data))
    }

    fn make_start_section(&mut self, data: Self::FuncIdx) -> Result<Self::Section, Self::Error> {
        if self.last_section_discrim > 7 {
            return Err(DefaultIRGeneratorError::InvalidSectionOrder(
                8,
                self.last_section_discrim,
            ));
        }
        self.last_section_discrim = 8;
        Ok(SectionType::Start(data))
    }

    fn make_element_section(
        &mut self,
        data: Box<[Self::Elem]>,
    ) -> Result<Self::Section, Self::Error> {
        if self.last_section_discrim > 8 {
            return Err(DefaultIRGeneratorError::InvalidSectionOrder(
                9,
                self.last_section_discrim,
            ));
        }
        self.max_valid_element_index = data.len() as u32;
        self.last_section_discrim = 9;
        Ok(SectionType::Element(data))
    }

    fn make_code_section(&mut self, code: Box<[Self::Code]>) -> Result<Self::Section, Self::Error> {
        if self.last_section_discrim > 9 {
            return Err(DefaultIRGeneratorError::InvalidSectionOrder(
                0xa,
                self.last_section_discrim,
            ));
        }
        self.max_valid_element_index = code.len() as u32;
        self.last_section_discrim = 0xa;
        Ok(SectionType::Code(code))
    }

    fn make_data_section(&mut self, data: Box<[Self::Data]>) -> Result<Self::Section, Self::Error> {
        if self.last_section_discrim > 0xa {
            return Err(DefaultIRGeneratorError::InvalidSectionOrder(
                0xb,
                self.last_section_discrim,
            ));
        }

        let data_len = data.len() as u32;
        if let Some(max_valid_data_index) = self.max_valid_data_index {
            if max_valid_data_index != data_len {
                return Err(DefaultIRGeneratorError::DatacountMismatch(
                    max_valid_data_index,
                    data_len,
                ));
            }
        } else {
            self.max_valid_data_index = Some(data_len);
        }
        self.last_section_discrim = 0xb;
        Ok(SectionType::Data(data))
    }

    // Datacount appears *before* code and data sections if present. It should not update the last
    // seen discriminant but it also cannot be repeated.
    fn make_datacount_section(&mut self, data: u32) -> Result<Self::Section, Self::Error> {
        if self.max_valid_data_index.is_some() || self.last_section_discrim > 0x9 {
            return Err(DefaultIRGeneratorError::InvalidSectionOrder(
                0x12,
                self.last_section_discrim,
            ));
        }
        self.max_valid_data_index = Some(data);
        Ok(SectionType::DataCount(data))
    }

    fn make_type_index(&mut self, candidate: u32) -> Result<Self::TypeIdx, Self::Error> {
        if candidate < self.max_valid_type_index {
            Ok(TypeIdx(candidate))
        } else {
            Err(DefaultIRGeneratorError::InvalidTypeIndex(
                candidate,
                self.max_valid_type_index,
            ))
        }
    }

    fn make_global_index(&mut self, candidate: u32) -> Result<Self::GlobalIdx, Self::Error> {
        if candidate < self.max_valid_global_index {
            Ok(GlobalIdx(candidate))
        } else {
            Err(DefaultIRGeneratorError::InvalidGlobalIndex(
                candidate,
                self.max_valid_global_index,
            ))
        }
    }

    fn make_table_index(&mut self, candidate: u32) -> Result<Self::TableIdx, Self::Error> {
        if candidate < self.max_valid_table_index {
            Ok(TableIdx(candidate))
        } else {
            Err(DefaultIRGeneratorError::InvalidTableIndex(
                candidate,
                self.max_valid_table_index,
            ))
        }
    }

    fn make_mem_index(&mut self, candidate: u32) -> Result<Self::MemIdx, Self::Error> {
        if candidate < self.max_valid_mem_index {
            Ok(MemIdx(candidate))
        } else {
            Err(DefaultIRGeneratorError::InvalidMemIndex(
                candidate,
                self.max_valid_mem_index,
            ))
        }
    }

    fn make_func_index(&mut self, candidate: u32) -> Result<Self::FuncIdx, Self::Error> {
        if candidate < self.max_valid_func_index {
            Ok(FuncIdx(candidate))
        } else {
            Err(DefaultIRGeneratorError::InvalidMemIndex(
                candidate,
                self.max_valid_func_index,
            ))
        }
    }

    fn make_local_index(&mut self, candidate: u32) -> Result<Self::LocalIdx, Self::Error> {
        if candidate as usize >= self.current_locals.len() {
            return Err(Self::Error::InvalidLocalIndex(
                candidate,
                self.current_locals.len() as u32,
            ));
        }
        Ok(LocalIdx(candidate))
    }

    fn make_func_type(
        &mut self,
        params: Option<Self::ResultType>,
        returns: Option<Self::ResultType>,
    ) -> Result<Self::Type, Self::Error> {
        Ok(Type(
            params.unwrap_or_default(),
            returns.unwrap_or_default(),
        ))
    }

    fn make_export_desc_func(
        &mut self,
        func_idx: Self::FuncIdx,
    ) -> Result<Self::ExportDesc, Self::Error> {
        Ok(ExportDesc::Func(func_idx))
    }

    fn make_export_desc_global(
        &mut self,
        global_idx: Self::GlobalIdx,
    ) -> Result<Self::ExportDesc, Self::Error> {
        Ok(ExportDesc::Global(global_idx))
    }

    fn make_export_desc_memtype(
        &mut self,
        mem_idx: Self::MemIdx,
    ) -> Result<Self::ExportDesc, Self::Error> {
        Ok(ExportDesc::Mem(mem_idx))
    }

    fn make_export_desc_table(
        &mut self,
        table_idx: Self::TableIdx,
    ) -> Result<Self::ExportDesc, Self::Error> {
        Ok(ExportDesc::Table(table_idx))
    }

    fn make_import_desc_func(
        &mut self,
        type_idx: Self::TypeIdx,
    ) -> Result<Self::ImportDesc, Self::Error> {
        self.max_valid_func_index += 1;
        Ok(ImportDesc::Func(type_idx))
    }

    fn make_import_desc_global(
        &mut self,
        global_type: Self::GlobalType,
    ) -> Result<Self::ImportDesc, Self::Error> {
        self.max_valid_global_index += 1;
        Ok(ImportDesc::Global(global_type))
    }

    fn make_import_desc_memtype(
        &mut self,
        mem_type: Self::Limits,
    ) -> Result<Self::ImportDesc, Self::Error> {
        self.max_valid_mem_index += 1;
        Ok(ImportDesc::Mem(MemType(mem_type)))
    }

    fn make_import_desc_table(
        &mut self,
        table_type: Self::TableType,
    ) -> Result<Self::ImportDesc, Self::Error> {
        self.max_valid_table_index += 1;
        Ok(ImportDesc::Table(table_type))
    }

    fn make_import(
        &mut self,
        modname: Self::Name,
        name: Self::Name,
        desc: Self::ImportDesc,
    ) -> Result<Self::Import, Self::Error> {
        Ok(Import {
            r#mod: modname,
            nm: name,
            desc,
        })
    }

    fn make_module(&mut self, sections: Vec<Self::Section>) -> Result<Self::Module, Self::Error> {
        let mut builder = ModuleBuilder::new();
        for section in sections {
            builder = match section {
                SectionType::Custom(xs) => builder.custom_section(xs),
                SectionType::Type(xs) => builder.type_section(xs),
                SectionType::Import(xs) => builder.import_section(xs),
                SectionType::Function(xs) => builder.function_section(xs),
                SectionType::Table(xs) => builder.table_section(xs),
                SectionType::Memory(xs) => builder.memory_section(xs),
                SectionType::Global(xs) => builder.global_section(xs),
                SectionType::Export(xs) => builder.export_section(xs),
                SectionType::Start(xs) => builder.start_section(xs),
                SectionType::Element(xs) => builder.element_section(xs),
                SectionType::Code(xs) => builder.code_section(xs),
                SectionType::Data(xs) => builder.data_section(xs),
                SectionType::DataCount(xs) => builder.datacount_section(xs),
            };
        }
        Ok(builder.build())
    }

    fn make_code(&mut self, item: Self::Func) -> Result<Self::Code, Self::Error> {
        Ok(Code(item))
    }

    fn make_data_active(
        &mut self,
        bytes: Box<[u8]>,
        mem_idx: Self::MemIdx,
        expr: Self::Expr,
    ) -> Result<Self::Data, Self::Error> {
        Ok(Data::Active(ByteVec(bytes), mem_idx, expr))
    }

    fn make_data_passive(&mut self, bytes: Box<[u8]>) -> Result<Self::Data, Self::Error> {
        Ok(Data::Passive(ByteVec(bytes)))
    }

    fn make_limits(&mut self, lower: u32, upper: Option<u32>) -> Result<Self::Limits, Self::Error> {
        Ok(if let Some(upper) = upper {
            Limits::Range(lower, upper)
        } else {
            Limits::Min(lower)
        })
    }

    fn make_export(
        &mut self,
        name: Self::Name,
        desc: Self::ExportDesc,
    ) -> Result<Self::Export, Self::Error> {
        Ok(Export { nm: name, desc })
    }

    fn make_expr(&mut self, instrs: Vec<Self::Instr>) -> Result<Self::Expr, Self::Error> {
        Ok(Expr(instrs))
    }

    fn start_func(&mut self) {
        self.next_func_idx += 1;
    }

    fn make_func(
        &mut self,
        locals: Box<[Self::Local]>,
        expr: Self::Expr,
    ) -> Result<Self::Func, Self::Error> {
        eprintln!("clearing locals");
        self.current_locals.clear();
        Ok(Func { locals, expr })
    }

    fn make_global(
        &mut self,
        global_type: Self::GlobalType,
        expr: Self::Expr,
    ) -> Result<Self::Global, Self::Error> {
        Ok(Global(global_type, expr))
    }

    fn make_local(
        &mut self,
        count: u32,
        val_type: Self::ValType,
    ) -> Result<Self::Local, Self::Error> {
        let local = Local(count, val_type);
        self.current_locals.extend((0..local.0).map(|_| local.1));
        eprintln!(
            "adding {count} locals for a grand total of {}",
            self.current_locals.len()
        );
        Ok(local)
    }

    fn make_instr_select(
        &mut self,
        items: Box<[Self::ValType]>,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        instrs.push(Instr::Select(items));
        Ok(())
    }

    fn make_instr_table(
        &mut self,
        items: Box<[u32]>,
        alternate: u32,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        let items = items.iter().map(|xs| LabelIdx(*xs)).collect();
        instrs.push(Instr::BrTable(items, LabelIdx(alternate)));
        Ok(())
    }

    fn make_instr_unary64(
        &mut self,
        code: u8,
        arg0: u64,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        match code {
            0x42 => instrs.push(Instr::I64Const(arg0 as i64)),
            0x44 => instrs.push(Instr::F64Const(f64::from_bits(arg0))),
            _ => return Err(DefaultIRGeneratorError::InvalidInstruction(code)),
        }
        Ok(())
    }

    fn make_instr_binary(
        &mut self,
        code: u8,
        arg0: u32,
        arg1: u32,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        instrs.push(match code {
            0x11 => Instr::CallIndirect(self.make_type_index(arg0)?, self.make_table_index(arg1)?),
            0x28 => Instr::I32Load(MemArg(arg0, arg1)),
            0x29 => Instr::I64Load(MemArg(arg0, arg1)),
            0x2a => Instr::F32Load(MemArg(arg0, arg1)),
            0x2b => Instr::F64Load(MemArg(arg0, arg1)),
            0x2c => Instr::I32Load8S(MemArg(arg0, arg1)),
            0x2d => Instr::I32Load8U(MemArg(arg0, arg1)),
            0x2e => Instr::I32Load16S(MemArg(arg0, arg1)),
            0x2f => Instr::I32Load16U(MemArg(arg0, arg1)),
            0x30 => Instr::I64Load8S(MemArg(arg0, arg1)),
            0x31 => Instr::I64Load8U(MemArg(arg0, arg1)),
            0x32 => Instr::I64Load16S(MemArg(arg0, arg1)),
            0x33 => Instr::I64Load16U(MemArg(arg0, arg1)),
            0x34 => Instr::I64Load32S(MemArg(arg0, arg1)),
            0x35 => Instr::I64Load32U(MemArg(arg0, arg1)),
            0x36 => Instr::I32Store(MemArg(arg0, arg1)),
            0x37 => Instr::I64Store(MemArg(arg0, arg1)),
            0x38 => Instr::F32Store(MemArg(arg0, arg1)),
            0x39 => Instr::F64Store(MemArg(arg0, arg1)),
            0x3a => Instr::I32Store8(MemArg(arg0, arg1)),
            0x3b => Instr::I32Store16(MemArg(arg0, arg1)),
            0x3c => Instr::I64Store8(MemArg(arg0, arg1)),
            0x3d => Instr::I64Store16(MemArg(arg0, arg1)),
            0x3e => Instr::I64Store32(MemArg(arg0, arg1)),
            unk => return Err(DefaultIRGeneratorError::InvalidInstruction(unk)),
        });
        Ok(())
    }

    fn make_instr_unary(
        &mut self,
        code: u8,
        arg0: u32,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        instrs.push(match code {
            0x10 => Instr::Call(self.make_func_index(arg0)?),
            0x41 => Instr::I32Const(arg0 as i32),
            0x43 => Instr::F32Const(f32::from_bits(arg0)),
            0x20 => Instr::LocalGet(self.make_local_index(arg0)?),
            0x21 => Instr::LocalSet(self.make_local_index(arg0)?),
            0x22 => Instr::LocalTee(self.make_local_index(arg0)?),
            0x23 => Instr::GlobalGet(self.make_global_index(arg0)?),
            0x24 => Instr::GlobalSet(self.make_global_index(arg0)?),
            0x25 => Instr::TableGet(self.make_table_index(arg0)?),
            0x26 => Instr::TableSet(self.make_table_index(arg0)?),
            0x3f => Instr::MemorySize(MemIdx(arg0)),
            0x40 => Instr::MemoryGrow(MemIdx(arg0)),
            0xd2 => Instr::RefFunc(self.make_func_index(arg0)?),
            // TODO: "make_label_idx"
            0x0c => Instr::Br(LabelIdx(arg0)),
            0x0d => Instr::BrIf(LabelIdx(arg0)),
            unk => return Err(DefaultIRGeneratorError::InvalidInstruction(unk)),
        });
        Ok(())
    }

    fn make_instr_nullary(
        &mut self,
        code: u8,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        instrs.push(match code {
            0x00 => Instr::Unreachable,
            0x01 => Instr::Nop,
            0xd1 => Instr::RefIsNull,
            0x1a => Instr::Drop,
            0x1b => Instr::SelectEmpty,
            0x0f => Instr::Return,
            0x45 => Instr::I32Eqz,
            0x46 => Instr::I32Eq,
            0x47 => Instr::I32Ne,
            0x48 => Instr::I32LtS,
            0x49 => Instr::I32LtU,
            0x4a => Instr::I32GtS,
            0x4b => Instr::I32GtU,
            0x4c => Instr::I32LeS,
            0x4d => Instr::I32LeU,
            0x4e => Instr::I32GeS,
            0x4f => Instr::I32GeU,
            0x50 => Instr::I64Eqz,
            0x51 => Instr::I64Eq,
            0x52 => Instr::I64Ne,
            0x53 => Instr::I64LtS,
            0x54 => Instr::I64LtU,
            0x55 => Instr::I64GtS,
            0x56 => Instr::I64GtU,
            0x57 => Instr::I64LeS,
            0x58 => Instr::I64LeU,
            0x59 => Instr::I64GeS,
            0x5a => Instr::I64GeU,
            0x5b => Instr::F32Eq,
            0x5c => Instr::F32Ne,
            0x5d => Instr::F32Lt,
            0x5e => Instr::F32Gt,
            0x5f => Instr::F32Le,
            0x60 => Instr::F32Ge,
            0x61 => Instr::F64Eq,
            0x62 => Instr::F64Ne,
            0x63 => Instr::F64Lt,
            0x64 => Instr::F64Gt,
            0x65 => Instr::F64Le,
            0x66 => Instr::F64Ge,
            0x67 => Instr::I32Clz,
            0x68 => Instr::I32Ctz,
            0x69 => Instr::I32Popcnt,
            0x6a => Instr::I32Add,
            0x6b => Instr::I32Sub,
            0x6c => Instr::I32Mul,
            0x6d => Instr::I32DivS,
            0x6e => Instr::I32DivU,
            0x6f => Instr::I32RemS,
            0x70 => Instr::I32RemU,
            0x71 => Instr::I32And,
            0x72 => Instr::I32Ior,
            0x73 => Instr::I32Xor,
            0x74 => Instr::I32Shl,
            0x75 => Instr::I32ShrS,
            0x76 => Instr::I32ShrU,
            0x77 => Instr::I32Rol,
            0x78 => Instr::I32Ror,
            0x79 => Instr::I64Clz,
            0x7a => Instr::I64Ctz,
            0x7b => Instr::I64Popcnt,
            0x7c => Instr::I64Add,
            0x7d => Instr::I64Sub,
            0x7e => Instr::I64Mul,
            0x7f => Instr::I64DivS,
            0x80 => Instr::I64DivU,
            0x81 => Instr::I64RemS,
            0x82 => Instr::I64RemU,
            0x83 => Instr::I64And,
            0x84 => Instr::I64Ior,
            0x85 => Instr::I64Xor,
            0x86 => Instr::I64Shl,
            0x87 => Instr::I64ShrS,
            0x88 => Instr::I64ShrU,
            0x89 => Instr::I64Rol,
            0x8a => Instr::I64Ror,
            0x8b => Instr::F32Abs,
            0x8c => Instr::F32Neg,
            0x8d => Instr::F32Ceil,
            0x8e => Instr::F32Floor,
            0x8f => Instr::F32Trunc,
            0x90 => Instr::F32NearestInt,
            0x91 => Instr::F32Sqrt,
            0x92 => Instr::F32Add,
            0x93 => Instr::F32Sub,
            0x94 => Instr::F32Mul,
            0x95 => Instr::F32Div,
            0x96 => Instr::F32Min,
            0x97 => Instr::F32Max,
            0x98 => Instr::F32CopySign,
            0x99 => Instr::F64Abs,
            0x9a => Instr::F64Neg,
            0x9b => Instr::F64Ceil,
            0x9c => Instr::F64Floor,
            0x9d => Instr::F64Trunc,
            0x9e => Instr::F64NearestInt,
            0x9f => Instr::F64Sqrt,
            0xa0 => Instr::F64Add,
            0xa1 => Instr::F64Sub,
            0xa2 => Instr::F64Mul,
            0xa3 => Instr::F64Div,
            0xa4 => Instr::F64Min,
            0xa5 => Instr::F64Max,
            0xa6 => Instr::F64CopySign,
            0xa7 => Instr::I32ConvertI64,
            0xa8 => Instr::I32SConvertF32,
            0xa9 => Instr::I32UConvertF32,
            0xaa => Instr::I32SConvertF64,
            0xab => Instr::I32UConvertF64,
            0xac => Instr::I64SConvertI32,
            0xad => Instr::I64UConvertI32,
            0xae => Instr::I64SConvertF32,
            0xaf => Instr::I64UConvertF32,
            0xb0 => Instr::I64SConvertF64,
            0xb1 => Instr::I64UConvertF64,
            0xb2 => Instr::F32SConvertI32,
            0xb3 => Instr::F32UConvertI32,
            0xb4 => Instr::F32SConvertI64,
            0xb5 => Instr::F32UConvertI64,
            0xb6 => Instr::F32ConvertF64,
            0xb7 => Instr::F64SConvertI32,
            0xb8 => Instr::F64UConvertI32,
            0xb9 => Instr::F64SConvertI64,
            0xba => Instr::F64UConvertI64,
            0xbb => Instr::F64ConvertF32,
            0xbc => Instr::I32ReinterpretF32,
            0xbd => Instr::I64ReinterpretF64,
            0xbe => Instr::F32ReinterpretI32,
            0xbf => Instr::F64ReinterpretI64,
            0xc0 => Instr::I32SExtendI8,
            0xc1 => Instr::I32SExtendI16,
            0xc2 => Instr::I64SExtendI8,
            0xc3 => Instr::I64SExtendI16,
            0xc4 => Instr::I64SExtendI32,
            unk => return Err(DefaultIRGeneratorError::InvalidInstruction(unk)),
        });
        Ok(())
    }

    fn make_instr_block(
        &mut self,
        block_kind: u8,
        block_type: Self::BlockType,
        expr: Self::Expr,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        match block_kind {
            0x02 => instrs.push(Instr::Block(block_type, expr.0.into_boxed_slice())),
            0x03 => instrs.push(Instr::Loop(block_type, expr.0.into_boxed_slice())),
            unk => return Err(DefaultIRGeneratorError::InvalidInstruction(unk)),
        }
        Ok(())
    }

    fn make_instr_block_ifelse(
        &mut self,
        block_type: Self::BlockType,
        consequent: Self::Expr,
        alternate: Option<Self::Expr>,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        instrs.push(if let Some(alternate) = alternate {
            Instr::IfElse(
                block_type,
                consequent.0.into_boxed_slice(),
                alternate.0.into_boxed_slice(),
            )
        } else {
            Instr::If(block_type, consequent.0.into_boxed_slice())
        });
        Ok(())
    }

    fn make_elem_from_indices(
        &mut self,
        kind: Option<u32>,
        mode: Self::ElemMode,
        idxs: Box<[u32]>,
        flags: u8,
    ) -> Result<Self::Elem, Self::Error> {
        let exprs = idxs
            .iter()
            .map(|xs| Ok(Expr(vec![Instr::RefFunc(self.make_func_index(*xs)?)])))
            .collect::<Result<Box<[_]>, Self::Error>>()?;

        let kind = kind.unwrap_or_default();
        if kind != 0 {
            // TODO: create a better error
            return Err(DefaultIRGeneratorError::InvalidRefType(kind as u8));
        }

        Ok(Elem {
            mode,
            kind: RefType::FuncRef,
            exprs,
            flags,
        })
    }

    fn make_elem_from_exprs(
        &mut self,
        kind: Option<Self::RefType>,
        mode: Self::ElemMode,
        exprs: Box<[Self::Expr]>,
        flags: u8,
    ) -> Result<Self::Elem, Self::Error> {
        Ok(Elem {
            mode,
            kind: kind.unwrap_or_default(),
            exprs,
            flags,
        })
    }

    fn make_elem_mode_passive(&mut self) -> Result<Self::ElemMode, Self::Error> {
        Ok(ElemMode::Passive)
    }

    fn make_elem_mode_declarative(&mut self) -> Result<Self::ElemMode, Self::Error> {
        Ok(ElemMode::Declarative)
    }

    fn make_elem_mode_active(
        &mut self,
        table_idx: Self::TableIdx,
        expr: Self::Expr,
    ) -> Result<Self::ElemMode, Self::Error> {
        if table_idx.0 >= self.max_valid_table_index {
            return Err(DefaultIRGeneratorError::InvalidTableIndex(
                table_idx.0,
                self.max_valid_table_index,
            ));
        }

        Ok(ElemMode::Active {
            table_idx,
            offset: expr,
        })
    }

    fn make_ref_type(&mut self, data: u8) -> Result<Self::RefType, Self::Error> {
        let ValType::RefType(ref_type) = self.make_val_type(data)? else {
            return Err(DefaultIRGeneratorError::InvalidRefType(data));
        };

        Ok(ref_type)
    }
}
