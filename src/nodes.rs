use std::fmt::Debug;

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct ByteVec<'a>(pub(crate) &'a [u8]);
#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Name<'a>(pub(crate) &'a str);

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum NumType {
    I32,
    I64,
    F32,
    F64,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum VecType {
    V128,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum RefType {
    FuncRef,
    ExternRef,
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum ValType {
    NumType(NumType),
    VecType(VecType),
    RefType(RefType),
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct ResultType(pub(crate) Vec<ValType>);

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Type(pub(crate) ResultType, pub(crate) ResultType);

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum Limits {
    Min(u32),
    Range(u32, u32),
}

impl Limits {
    pub(crate) fn min(&self) -> u32 {
        *match self {
            Limits::Min(min) => min,
            Limits::Range(min, _) => min,
        }
    }

    pub(crate) fn max(&self) -> Option<u32> {
        match self {
            Limits::Min(_) => None,
            Limits::Range(_, max) => Some(*max),
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) struct MemType(pub(crate) Limits);

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) struct TableType(pub(crate) RefType, pub(crate) Limits);

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum Mutability {
    Const,
    Variable,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) struct GlobalType(pub(crate) ValType, pub(crate) Mutability);

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Global(pub(crate) GlobalType, pub(crate) Expr);

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum BlockType {
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
pub(crate) struct MemArg(pub(crate) u32, pub(crate) u32);

impl MemArg {
    pub(crate) fn memidx(&self) -> usize {
        0
    }

    pub(crate) fn offset(&self) -> usize {
        self.1 as usize
    }

    pub(crate) fn align(&self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Instr {
    // Control Instructions
    Unreachable,
    Nop,
    Block(BlockType, Vec<Instr>),
    Loop(BlockType, Vec<Instr>),
    If(BlockType, Vec<Instr>),
    IfElse(BlockType, Vec<Instr>, Vec<Instr>),
    Br(LabelIdx),
    BrIf(LabelIdx),
    BrTable(Vec<LabelIdx>, LabelIdx),
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
    Select(Vec<ValType>),

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
pub(crate) struct Expr(pub(crate) Vec<Instr>);

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) struct TypeIdx(pub(crate) u32);

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) struct FuncIdx(pub(crate) u32);

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) struct CodeIdx(pub(crate) u32);

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) struct TableIdx(pub(crate) u32);

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) struct MemIdx(pub(crate) u32);

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) struct GlobalIdx(pub(crate) u32);

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) struct ElemIdx(pub(crate) u32);

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) struct DataIdx(pub(crate) u32);

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) struct LocalIdx(pub(crate) u32);

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) struct LabelIdx(pub(crate) u32);

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Import<'a> {
    pub(crate) r#mod: Name<'a>,
    pub(crate) nm: Name<'a>,
    pub(crate) desc: ImportDesc,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum ImportDesc {
    Func(TypeIdx),
    Table(TableType),
    Mem(MemType),
    Global(GlobalType),
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Export<'a> {
    pub(crate) nm: Name<'a>,
    pub(crate) desc: ExportDesc,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum ExportDesc {
    Func(FuncIdx),
    Table(TableIdx),
    Mem(MemIdx),
    Global(GlobalIdx),
}

// TODO: Woof, this is a bad type. Parsed elements are much more straightforward than their
// bit-flagged binary representation.
#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Elem {
    ActiveSegmentFuncs(Expr, Vec<FuncIdx>),
    PassiveSegment(u32, Vec<FuncIdx>),
    ActiveSegment(TableIdx, Expr, u32, Vec<FuncIdx>),
    DeclarativeSegment(u32, Vec<FuncIdx>),

    ActiveSegmentExpr(Expr, Vec<Expr>),
    PassiveSegmentExpr(RefType, Vec<Expr>),
    ActiveSegmentTableAndExpr(TableIdx, Expr, RefType, Vec<Expr>),
    DeclarativeSegmentExpr(RefType, Vec<Expr>),
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) struct Local(pub(crate) u32, pub(crate) ValType);

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Func {
    pub(crate) locals: Vec<Local>,
    pub(crate) expr: Expr,
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Code(pub(crate) Func);

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Data<'a> {
    Active(ByteVec<'a>, MemIdx, Expr),
    Passive(ByteVec<'a>),
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum SectionType<'a> {
    Custom(&'a [u8]),
    Type(Vec<Type>),
    Import(Vec<Import<'a>>),
    Function(Vec<TypeIdx>),
    Table(Vec<TableType>),
    Memory(Vec<MemType>),
    Global(Vec<Global>),
    Export(Vec<Export<'a>>),
    Start(FuncIdx),
    Element(Vec<Elem>),
    Code(Vec<Code>),
    Data(Vec<Data<'a>>),
    DataCount(u32),
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Section<T: Debug + PartialEq + Clone> {
    pub(crate) index: usize,
    pub(crate) inner: T,
}

#[derive(Debug, PartialEq, Clone, Default)]
pub(crate) struct Module<'a> {
    pub(crate) custom_sections: Vec<Section<&'a [u8]>>,
    pub(crate) type_section: Option<Section<Vec<Type>>>,
    pub(crate) import_section: Option<Section<Vec<Import<'a>>>>,
    pub(crate) function_section: Option<Section<Vec<TypeIdx>>>,
    pub(crate) table_section: Option<Section<Vec<TableType>>>,
    pub(crate) memory_section: Option<Section<Vec<MemType>>>,
    pub(crate) global_section: Option<Section<Vec<Global>>>,
    pub(crate) export_section: Option<Section<Vec<Export<'a>>>>,
    pub(crate) start_section: Option<Section<FuncIdx>>,
    pub(crate) element_section: Option<Section<Vec<Elem>>>,
    pub(crate) code_section: Option<Section<Vec<Code>>>,
    pub(crate) data_section: Option<Section<Vec<Data<'a>>>>,
    pub(crate) datacount_section: Option<Section<u32>>,
}

pub struct ModuleBuilder<'a> {
    inner: Module<'a>,
    index: usize,
}

impl<'a> ModuleBuilder<'a> {
    pub(crate) fn new() -> Self {
        Self {
            inner: Default::default(),
            index: 0,
        }
    }

    pub(crate) fn custom_section(mut self, v: &'a [u8]) -> Self {
        self.inner.custom_sections.push(Section {
            inner: v,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub(crate) fn type_section(mut self, xs: Vec<Type>) -> Self {
        self.inner.type_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub(crate) fn import_section(mut self, xs: Vec<Import<'a>>) -> Self {
        self.inner.import_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub(crate) fn function_section(mut self, xs: Vec<TypeIdx>) -> Self {
        self.inner.function_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub(crate) fn table_section(mut self, xs: Vec<TableType>) -> Self {
        self.inner.table_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub(crate) fn memory_section(mut self, xs: Vec<MemType>) -> Self {
        self.inner.memory_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub(crate) fn global_section(mut self, xs: Vec<Global>) -> Self {
        self.inner.global_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub(crate) fn export_section(mut self, xs: Vec<Export<'a>>) -> Self {
        self.inner.export_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub(crate) fn start_section(mut self, xs: FuncIdx) -> Self {
        self.inner.start_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub(crate) fn element_section(mut self, xs: Vec<Elem>) -> Self {
        self.inner.element_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub(crate) fn code_section(mut self, xs: Vec<Code>) -> Self {
        self.inner.code_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub(crate) fn data_section(mut self, xs: Vec<Data<'a>>) -> Self {
        self.inner.data_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }
    pub(crate) fn datacount_section(mut self, xs: u32) -> Self {
        self.inner.datacount_section.replace(Section {
            inner: xs,
            index: self.index,
        });
        self.index += 1;
        self
    }

    pub(crate) fn build(self) -> Module<'a> {
        self.inner
    }
}

impl<'a> Module<'a> {
    pub(crate) fn custom_sections(&self) -> impl Iterator<Item = &[u8]> {
        self.custom_sections.iter().map(|xs| xs.inner)
    }

    pub(crate) fn type_section(&self) -> Option<&[Type]> {
        self.type_section.as_ref().map(|xs| xs.inner.as_slice())
    }

    pub(crate) fn import_section(&self) -> Option<&[Import]> {
        self.import_section.as_ref().map(|xs| xs.inner.as_slice())
    }

    pub(crate) fn function_section(&self) -> Option<&[TypeIdx]> {
        self.function_section.as_ref().map(|xs| xs.inner.as_slice())
    }

    pub(crate) fn table_section(&self) -> Option<&[TableType]> {
        self.table_section.as_ref().map(|xs| xs.inner.as_slice())
    }

    pub(crate) fn memory_section(&self) -> Option<&[MemType]> {
        self.memory_section.as_ref().map(|xs| xs.inner.as_slice())
    }

    pub(crate) fn global_section(&self) -> Option<&[Global]> {
        self.global_section.as_ref().map(|xs| xs.inner.as_slice())
    }

    pub(crate) fn export_section(&self) -> Option<&[Export]> {
        self.export_section.as_ref().map(|xs| xs.inner.as_slice())
    }

    pub(crate) fn start_section(&self) -> Option<FuncIdx> {
        self.start_section.as_ref().map(|xs| xs.inner)
    }

    pub(crate) fn element_section(&self) -> Option<&[Elem]> {
        self.element_section.as_ref().map(|xs| xs.inner.as_slice())
    }

    pub(crate) fn code_section(&self) -> Option<&[Code]> {
        self.code_section.as_ref().map(|xs| xs.inner.as_slice())
    }
    pub(crate) fn data_section(&self) -> Option<&[Data]> {
        self.data_section.as_ref().map(|xs| xs.inner.as_slice())
    }

    pub(crate) fn datacount_section(&self) -> Option<u32> {
        self.datacount_section.as_ref().map(|xs| xs.inner)
    }
}
