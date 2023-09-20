#[derive(Debug, PartialEq, Clone)]
pub(crate) struct ByteVec(pub(crate) Vec<u8>);
#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Name(pub(crate) String);

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
pub(crate) struct FuncType(pub(crate) ResultType, pub(crate) ResultType);

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

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum BlockType {
    Empty,
    Val(ValType),
    TypeIndex(i32),
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) struct MemArg(pub(crate) u32, pub(crate) u32);

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
    GlobalGet(LocalIdx),
    GlobalSet(LocalIdx),

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
pub(crate) struct Import {
    pub(crate) r#mod: Name,
    pub(crate) nm: Name,
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
pub(crate) struct Export {
    pub(crate) nm: Name,
    pub(crate) desc: ExportDesc,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum ExportDesc {
    Func(TypeIdx),
    Table(TableType),
    Mem(MemType),
    Global(GlobalType),
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
pub(crate) enum Data {
    Active(ByteVec, MemIdx, Expr),
    Passive(ByteVec),
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Section {
    Custom(Vec<u8>),
    Type(Vec<FuncType>),
    Import(Vec<Import>),
    Function(Vec<TypeIdx>),
    Table(Vec<TableType>),
    Memory(Vec<MemType>),
    Global(GlobalType, Expr),
    Export(Vec<Export>),
    Start(FuncIdx),
    Element(Vec<Elem>),
    Code(Vec<Code>),
    Data(Vec<Data>),
    DataCount(u32),
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Module {
    pub(crate) sections: Vec<Section>,
}

impl Module {
    pub(crate) fn custom_sections(&self) -> impl Iterator<Item = &[u8]> {
        self.sections.iter().filter_map(|xs| if let Section::Custom(xs) = xs {
            Some(xs.as_slice())
        } else {
            None
        })
    }

    pub(crate) fn type_section(&self) -> Option<&[FuncType]> {
        self.sections.iter().filter_map(|xs| if let Section::Type(xs) = xs {
            Some(xs.as_slice())
        } else {
            None
        }).next()
    }

    pub(crate) fn import_section(&self) -> Option<&[Import]> {
        self.sections.iter().filter_map(|xs| if let Section::Import(xs) = xs {
            Some(xs.as_slice())
        } else {
            None
        }).next()
    }

    pub(crate) fn function_section(&self) -> Option<&[TypeIdx]> {
        self.sections.iter().filter_map(|xs| if let Section::Function(xs) = xs {
            Some(xs.as_slice())
        } else {
            None
        }).next()
    }

    pub(crate) fn table_section(&self) -> Option<&[TableType]> {
        self.sections.iter().filter_map(|xs| if let Section::Table(xs) = xs {
            Some(xs.as_slice())
        } else {
            None
        }).next()
    }

    pub(crate) fn memory_section(&self) -> Option<&[MemType]> {
        self.sections.iter().filter_map(|xs| if let Section::Memory(xs) = xs {
            Some(xs.as_slice())
        } else {
            None
        }).next()
    }

    pub(crate) fn global_section(&self) -> Option<(&GlobalType, &Expr)> {
        self.sections.iter().filter_map(|xs| if let Section::Global(xs, ys) = xs {
            Some((xs, ys))
        } else {
            None
        }).next()
    }

    pub(crate) fn export_section(&self) -> Option<&[Export]> {
        self.sections.iter().filter_map(|xs| if let Section::Export(xs) = xs {
            Some(xs.as_slice())
        } else {
            None
        }).next()
    }

    pub(crate) fn start_section(&self) -> Option<FuncIdx> {
        self.sections.iter().filter_map(|xs| if let Section::Start(xs) = xs {
            Some(*xs)
        } else {
            None
        }).next()
    }

    pub(crate) fn element_section(&self) -> Option<&[Elem]> {
        self.sections.iter().filter_map(|xs| if let Section::Element(xs) = xs {
            Some(xs.as_slice())
        } else {
            None
        }).next()
    }

    pub(crate) fn code_section(&self) -> Option<&[Code]> {
        self.sections.iter().filter_map(|xs| if let Section::Code(xs) = xs {
            Some(xs.as_slice())
        } else {
            None
        }).next()
    }

    pub(crate) fn data_section(&self) -> Option<&[Data]> {
        self.sections.iter().filter_map(|xs| if let Section::Data(xs) = xs {
            Some(xs.as_slice())
        } else {
            None
        }).next()
    }

    pub(crate) fn datacount_section(&self) -> Option<u32> {
        self.sections.iter().filter_map(|xs| if let Section::DataCount(xs) = xs {
            Some(*xs)
        } else {
            None
        }).next()
    }
}
