use std::fmt::Debug;

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
    Never,
}

impl std::fmt::Display for ValType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValType::NumType(NumType::I32) => f.write_str("i32"),
            ValType::NumType(NumType::F32) => f.write_str("f32"),
            ValType::NumType(NumType::I64) => f.write_str("i64"),
            ValType::NumType(NumType::F64) => f.write_str("f64"),
            ValType::VecType(VecType::V128) => f.write_str("v128"),
            ValType::RefType(RefType::FuncRef) => f.write_str("funcref"),
            ValType::RefType(RefType::ExternRef) => f.write_str("externref"),
            ValType::Never => f.write_str("!"),
        }
    }
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
/// A memarg comprises two elements: an alignment and an offset.
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
    CallIntrinsic(TypeIdx, usize),

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
    DropEmpty,
    Drop(ValType),
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
    F32Const(u32),
    F64Const(u64),
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
    pub fn is_empty(&self) -> bool {
        self.exprs.is_empty()
    }

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
#[non_exhaustive]
pub enum SectionType {
    Custom(String, Box<[u8]>),
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
    pub(crate) custom_sections: Vec<Section<(String, Box<[u8]>)>>,
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
    pub custom_sections: Vec<Section<(String, Box<[u8]>)>>,
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

    pub fn custom_section(mut self, name: String, v: Box<[u8]>) -> Self {
        self.inner.custom_sections.push(Section {
            inner: (name, v),
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

    pub fn custom_sections(&self) -> impl Iterator<Item = (&str, &[u8])> {
        self.custom_sections
            .iter()
            .map(|xs| (xs.inner.0.as_str(), &*xs.inner.1))
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
