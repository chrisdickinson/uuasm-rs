use crate::{
    defs::*,
    typechecker::{BlockKind, TypeChecker, TypeError},
    IR,
};
use thiserror::Error;

use std::{collections::HashSet, fmt::Debug};

#[derive(Clone, Debug, Error)]
pub enum DefaultIRGeneratorError {
    #[error("malformed UTF-8 encoding: {0}")]
    InvalidName(#[from] std::str::Utf8Error),

    #[error("Invalid instruction (got {0:X}H {1:X}H)")]
    InvalidInstruction(u8, u32),

    #[error("Invalid type (got {0:X}H)")]
    InvalidType(u8),

    #[error("unknown type index (got {0}; max is {1})")]
    InvalidTypeIndex(u32, u32),

    #[error("unknown global index (got {0}; max is {1})")]
    InvalidGlobalIndex(u32, u32),

    #[error("unknown table index (got {0}; max is {1})")]
    InvalidTableIndex(u32, u32),

    #[error("unknown local index (got {0}; max is {1})")]
    InvalidLocalIndex(u32, u32),

    #[error("unknown function {0} (max is {1})")]
    InvalidFuncIndex(u32, u32),

    #[error("label out of range (got {0}; max is {1})")]
    InvalidLabelIndex(u32, u32),

    #[error(
        "undeclared function reference (id {0} is not declared in an element, export, or import)"
    )]
    UndeclaredFuncIndex(u32),

    #[error("unknown memory {0} (max index is {1})")]
    InvalidMemIndex(u32, u32),

    #[error("unknown data index (got {0}; max is {1})")]
    InvalidDataIndex(u32, u32),

    #[error("unknown element index (got {0}; max is {1})")]
    InvalidElementIndex(u32, u32),

    #[error(
        "Invalid memory lower bound: size minimum must not be greater than maximum, got {0} {1}"
    )]
    MemoryBoundInvalid(u32, u32),

    #[error(
        "Invalid memory lower bound: memory size must be at most 65536 pages (4GiB), got {0} pages"
    )]
    MemoryLowerBoundTooLarge(u32),

    #[error(
        "Invalid memory upper bound: memory size must be at most 65536 pages (4GiB), got {0} pages"
    )]
    MemoryUpperBoundTooLarge(u32),

    #[error("Types out of order (got section type {0} after type {1})")]
    InvalidSectionOrder(u32, u32),

    #[error("Datacount section value did not match data element count (expected {0}, got {1})")]
    DatacountMismatch(u32, u32),

    #[error("Invalid reference type {0}")]
    InvalidRefType(u8),

    #[error("Invalid memory: multiple memories are not enabled")]
    MultimemoryDisabled,

    #[error("unexpected end of custom section")]
    IncompleteCustomSectionName,

    #[error("{0}")]
    TypeError(#[from] TypeError),
}

#[derive(Default, Clone, Debug)]
pub struct Features {
    enable_multiple_memories: bool,
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

    // constant expressions may only use imported globals
    global_import_boundary_idx: u32,

    next_func_idx: u32,

    type_checker: TypeChecker,

    types: Option<Box<[Type]>>,
    func_types: Option<Box<[TypeIdx]>>,
    table_types: Option<Box<[TableType]>>,

    global_types: Vec<GlobalType>,
    current_locals: Vec<ValType>,

    current_section_id: u8,
    valid_function_indices: HashSet<u32>,
    features: Features,
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
        self.type_checker.clear();
        let global_type = GlobalType(
            valtype,
            if is_mutable {
                Mutability::Variable
            } else {
                Mutability::Const
            },
        );
        self.global_types.push(global_type);
        Ok(global_type)
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
        let min = limits.min();
        let max = limits.max().unwrap_or_else(|| limits.min());
        if min > 0x10000 {
            return Err(Self::Error::MemoryLowerBoundTooLarge(min));
        }

        if max > 0x10000 {
            return Err(Self::Error::MemoryUpperBoundTooLarge(max));
        }

        if min > max {
            return Err(Self::Error::MemoryBoundInvalid(min, max));
        }

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
        let mut offset = 0;
        let mut shift = 0;
        let mut repr = 0;
        while {
            let Some(next) = data.get(offset) else {
                return Err(Self::Error::IncompleteCustomSectionName);
            };

            repr |= ((next & 0x7f) as u64) << shift;
            offset += 1;
            shift += 7;

            next & 0x80 != 0
        } {}

        if repr as usize + offset > data.len() {
            return Err(Self::Error::MultimemoryDisabled);
        }

        std::str::from_utf8(&data[offset..offset + repr as usize])?;

        Ok(SectionType::Custom(data))
    }

    fn make_type_section(&mut self, data: Box<[Type]>) -> Result<Self::Section, Self::Error> {
        if self.last_section_discrim > 0 {
            return Err(DefaultIRGeneratorError::InvalidSectionOrder(
                1,
                self.last_section_discrim,
            ));
        }

        self.types = Some(data.clone());
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
        self.func_types = Some(data.clone());
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
        self.table_types = Some(data.clone());
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
        if self.max_valid_mem_index > 1 && !self.features.enable_multiple_memories {
            return Err(Self::Error::MultimemoryDisabled);
        }

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
            if self.current_section_id != 0x08 {
                self.valid_function_indices.insert(candidate);
            }

            Ok(FuncIdx(candidate))
        } else {
            Err(DefaultIRGeneratorError::InvalidFuncIndex(
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

    fn make_data_index(&mut self, candidate: u32) -> Result<Self::DataIdx, Self::Error> {
        let data_count = self.max_valid_data_index.unwrap_or_default();
        if candidate > data_count {
            return Err(Self::Error::InvalidDataIndex(candidate, data_count));
        }
        Ok(DataIdx(candidate))
    }

    fn make_elem_index(&mut self, candidate: u32) -> Result<Self::ElemIdx, Self::Error> {
        if candidate > self.max_valid_element_index {
            return Err(Self::Error::InvalidElementIndex(
                candidate,
                self.max_valid_element_index,
            ));
        }
        Ok(ElemIdx(candidate))
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
        self.global_import_boundary_idx = self.max_valid_global_index;
        Ok(ImportDesc::Global(global_type))
    }

    fn make_import_desc_memtype(
        &mut self,
        mem_type: Self::Limits,
    ) -> Result<Self::ImportDesc, Self::Error> {
        self.max_valid_mem_index += 1;
        if self.max_valid_mem_index > 1 && !self.features.enable_multiple_memories {
            return Err(Self::Error::MultimemoryDisabled);
        }
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
        self.type_checker.pop_ctrl()?;
        self.type_checker.clear();
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

    fn start_block(&mut self, block_type: &BlockType) -> Result<(), Self::Error> {
        let (inputs, outputs) = self.block_type(block_type);

        self.type_checker
            .pop_vals(&inputs)
            .map_err(DefaultIRGeneratorError::TypeError)?;
        self.type_checker
            .push_ctrl(BlockKind::Block, inputs, outputs);
        Ok(())
    }

    fn start_loop(&mut self, block_type: &BlockType) -> Result<(), Self::Error> {
        let (inputs, outputs) = self.block_type(block_type);

        self.type_checker
            .pop_vals(&inputs)
            .map_err(DefaultIRGeneratorError::TypeError)?;
        self.type_checker
            .push_ctrl(BlockKind::Loop, inputs, outputs);
        Ok(())
    }

    fn start_if(&mut self, block_type: &BlockType) -> Result<(), Self::Error> {
        let (inputs, outputs) = self.block_type(block_type);
        #[cfg(any())]
        eprintln!("start_if inputs={inputs:?} outputs={outputs:?}");

        self.type_checker
            .pop_val(Some(ValType::NumType(NumType::I32).into()))
            .map_err(DefaultIRGeneratorError::TypeError)?;
        self.type_checker
            .pop_vals(&inputs)
            .map_err(DefaultIRGeneratorError::TypeError)?;
        self.type_checker.push_ctrl(BlockKind::If, inputs, outputs);
        Ok(())
    }

    fn start_else(&mut self, _: &BlockType) -> Result<(), Self::Error> {
        let frame = self.type_checker.pop_ctrl()?;
        #[cfg(any())]
        eprintln!("start_else");
        self.type_checker
            .push_ctrl(BlockKind::Else, frame.start_types, frame.end_types);

        Ok(())
    }

    fn start_section(&mut self, section_id: u8, _section_size: u32) -> Result<(), Self::Error> {
        self.current_section_id = section_id;
        Ok(())
    }

    fn start_func(&mut self) -> Result<(), Self::Error> {
        #[cfg(any())]
        eprintln!(
            "= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = "
        );
        self.type_checker.clear();
        let (start, end) = self.func_type(&FuncIdx(self.next_func_idx));

        self.current_locals.extend(start.iter().cloned());
        self.type_checker
            .push_ctrl(BlockKind::Func, Box::new([]), end);
        self.next_func_idx += 1;

        Ok(())
    }

    fn start_elem_active_table_index(&mut self) -> Result<(), Self::Error> {
        self.type_checker.push_ctrl(
            BlockKind::ConstantExpression,
            Box::new([]),
            Box::new([ValType::NumType(NumType::I32)]),
        );

        Ok(())
    }

    fn start_elem_reftype_list(
        &mut self,
        ref_type: Option<&Self::RefType>,
    ) -> Result<(), Self::Error> {
        self.type_checker.push_ctrl(
            BlockKind::ConstantExpression,
            Box::new([]),
            Box::new([ValType::RefType(ref_type.copied().unwrap_or_default())]),
        );

        Ok(())
    }

    fn start_data_offset(&mut self) -> Result<(), Self::Error> {
        self.type_checker.push_ctrl(
            BlockKind::ConstantExpression,
            Box::new([]),
            Box::new([ValType::NumType(NumType::I32)]),
        );

        Ok(())
    }

    fn start_global(&mut self, global_type: &Self::GlobalType) -> Result<(), Self::Error> {
        self.type_checker.push_ctrl(
            BlockKind::ConstantExpression,
            Box::new([]),
            Box::new([global_type.0]),
        );
        Ok(())
    }

    fn make_func(
        &mut self,
        locals: Box<[Self::Local]>,
        expr: Self::Expr,
    ) -> Result<Self::Func, Self::Error> {
        self.type_checker.pop_ctrl()?;
        self.type_checker.clear();
        // TODO: type tracer: check that the stack validates against the desired type of the
        // function
        self.current_locals.clear();
        Ok(Func { locals, expr })
    }

    fn make_global(
        &mut self,
        global_type: Self::GlobalType,
        expr: Self::Expr,
    ) -> Result<Self::Global, Self::Error> {
        self.type_checker.pop_ctrl()?;
        self.type_checker.clear();
        Ok(Global(global_type, expr))
    }

    fn make_local(
        &mut self,
        count: u32,
        val_type: Self::ValType,
    ) -> Result<Self::Local, Self::Error> {
        let local = Local(count, val_type);
        self.current_locals.extend((0..local.0).map(|_| local.1));
        Ok(local)
    }

    fn make_instr_select(
        &mut self,
        items: Box<[Self::ValType]>,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        instrs.push(self.trace(Instr::Select(items))?);
        Ok(())
    }

    fn make_instr_table(
        &mut self,
        items: &[u32],
        alternate: u32,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        let items = items.iter().map(|xs| LabelIdx(*xs)).collect();
        instrs.push(self.trace(Instr::BrTable(items, LabelIdx(alternate)))?);
        Ok(())
    }

    fn make_instr_arity1_64(
        &mut self,
        code: u8,
        subcode: u32,
        arg0: u64,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        instrs.push(self.trace(match (code, subcode) {
            (0x42, 0) => Instr::I64Const(arg0 as i64),
            (0x44, 0) => Instr::F64Const(f64::from_bits(arg0)),
            _ => return Err(DefaultIRGeneratorError::InvalidInstruction(code, subcode)),
        })?);
        Ok(())
    }

    fn make_instr_arity2(
        &mut self,
        code: u8,
        subcode: u32,
        arg0: u32,
        arg1: u32,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        let instr = match (code, subcode) {
            (0x11, 0) => {
                Instr::CallIndirect(self.make_type_index(arg0)?, self.make_table_index(arg1)?)
            }
            (0x28..=0x3e, 0) if self.max_valid_mem_index == 0 => {
                return Err(Self::Error::InvalidMemIndex(0, 0))
            }
            (0x28, 0) => Instr::I32Load(MemArg(arg0, arg1)),
            (0x29, 0) => Instr::I64Load(MemArg(arg0, arg1)),
            (0x2a, 0) => Instr::F32Load(MemArg(arg0, arg1)),
            (0x2b, 0) => Instr::F64Load(MemArg(arg0, arg1)),
            (0x2c, 0) => Instr::I32Load8S(MemArg(arg0, arg1)),
            (0x2d, 0) => Instr::I32Load8U(MemArg(arg0, arg1)),
            (0x2e, 0) => Instr::I32Load16S(MemArg(arg0, arg1)),
            (0x2f, 0) => Instr::I32Load16U(MemArg(arg0, arg1)),
            (0x30, 0) => Instr::I64Load8S(MemArg(arg0, arg1)),
            (0x31, 0) => Instr::I64Load8U(MemArg(arg0, arg1)),
            (0x32, 0) => Instr::I64Load16S(MemArg(arg0, arg1)),
            (0x33, 0) => Instr::I64Load16U(MemArg(arg0, arg1)),
            (0x34, 0) => Instr::I64Load32S(MemArg(arg0, arg1)),
            (0x35, 0) => Instr::I64Load32U(MemArg(arg0, arg1)),
            (0x36, 0) => Instr::I32Store(MemArg(arg0, arg1)),
            (0x37, 0) => Instr::I64Store(MemArg(arg0, arg1)),
            (0x38, 0) => Instr::F32Store(MemArg(arg0, arg1)),
            (0x39, 0) => Instr::F64Store(MemArg(arg0, arg1)),
            (0x3a, 0) => Instr::I32Store8(MemArg(arg0, arg1)),
            (0x3b, 0) => Instr::I32Store16(MemArg(arg0, arg1)),
            (0x3c, 0) => Instr::I64Store8(MemArg(arg0, arg1)),
            (0x3d, 0) => Instr::I64Store16(MemArg(arg0, arg1)),
            (0x3e, 0) => Instr::I64Store32(MemArg(arg0, arg1)),

            (0xfc, 0x08) => {
                Instr::MemoryInit(self.make_data_index(arg0)?, self.make_mem_index(arg1)?)
            }
            (0xfc, 0x0a) => {
                Instr::MemoryCopy(self.make_mem_index(arg0)?, self.make_mem_index(arg1)?)
            }
            (0xfc, 0x0c) => {
                Instr::TableInit(self.make_elem_index(arg0)?, self.make_table_index(arg1)?)
            }
            (0xfc, 0x0e) => {
                Instr::TableCopy(self.make_table_index(arg0)?, self.make_table_index(arg1)?)
            }

            _ => return Err(DefaultIRGeneratorError::InvalidInstruction(code, subcode)),
        };
        instrs.push(self.trace(instr)?);
        Ok(())
    }

    fn make_instr_arity1(
        &mut self,
        code: u8,
        subcode: u32,
        arg0: u32,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        let instr = match (code, subcode) {
            (0x10, 0) => Instr::Call(self.make_func_index(arg0)?),
            (0x41, 0) => Instr::I32Const(arg0 as i32),
            (0x43, 0) => Instr::F32Const(f32::from_bits(arg0)),
            (0x20, 0) => Instr::LocalGet(self.make_local_index(arg0)?),
            (0x21, 0) => Instr::LocalSet(self.make_local_index(arg0)?),
            (0x22, 0) => Instr::LocalTee(self.make_local_index(arg0)?),
            (0x23, 0) => Instr::GlobalGet(self.make_global_index(arg0)?),
            (0x24, 0) => Instr::GlobalSet(self.make_global_index(arg0)?),
            (0x25, 0) => Instr::TableGet(self.make_table_index(arg0)?),
            (0x26, 0) => Instr::TableSet(self.make_table_index(arg0)?),
            (0x3f..=0x40, 0) if self.max_valid_mem_index == 0 => {
                return Err(Self::Error::InvalidMemIndex(0, 0))
            }
            (0x3f, 0) => Instr::MemorySize(MemIdx(arg0)),
            (0x40, 0) => Instr::MemoryGrow(MemIdx(arg0)),
            (0xd0, 0) => Instr::RefNull(self.make_ref_type(arg0 as u8)?),
            (0xd2, 0) => {
                // If we're in the code section, validate the function index
                // against "Context.Refs"
                if self.current_section_id == 0x0a && !self.valid_function_indices.contains(&arg0) {
                    return Err(Self::Error::UndeclaredFuncIndex(arg0));
                }

                Instr::RefFunc(self.make_func_index(arg0)?)
            }
            // TODO: "make_label_idx"
            (0x0c, 0) => Instr::Br(LabelIdx(arg0)),
            (0x0d, 0) => Instr::BrIf(LabelIdx(arg0)),

            (0xfc, 0x09) => Instr::DataDrop(self.make_data_index(arg0)?),
            (0xfc, 0x0b) => Instr::MemoryFill(self.make_mem_index(arg0)?),
            (0xfc, 0x0d) => Instr::ElemDrop(self.make_elem_index(arg0)?),
            (0xfc, 0x0f) => Instr::TableGrow(self.make_table_index(arg0)?),
            (0xfc, 0x10) => Instr::TableSize(self.make_table_index(arg0)?),
            (0xfc, 0x11) => Instr::TableFill(self.make_table_index(arg0)?),
            _ => return Err(DefaultIRGeneratorError::InvalidInstruction(code, subcode)),
        };
        instrs.push(self.trace(instr)?);
        Ok(())
    }

    fn make_instr_arity0(
        &mut self,
        code: u8,
        subcode: u32,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        instrs.push(self.trace(match (code, subcode) {
            (0x00, 0) => Instr::Unreachable,
            (0x01, 0) => Instr::Nop,
            (0xd1, 0) => Instr::RefIsNull,
            (0x1a, 0) => Instr::Drop,
            (0x1b, 0) => Instr::SelectEmpty,
            (0x0f, 0) => Instr::Return,
            (0x45, 0) => Instr::I32Eqz,
            (0x46, 0) => Instr::I32Eq,
            (0x47, 0) => Instr::I32Ne,
            (0x48, 0) => Instr::I32LtS,
            (0x49, 0) => Instr::I32LtU,
            (0x4a, 0) => Instr::I32GtS,
            (0x4b, 0) => Instr::I32GtU,
            (0x4c, 0) => Instr::I32LeS,
            (0x4d, 0) => Instr::I32LeU,
            (0x4e, 0) => Instr::I32GeS,
            (0x4f, 0) => Instr::I32GeU,
            (0x50, 0) => Instr::I64Eqz,
            (0x51, 0) => Instr::I64Eq,
            (0x52, 0) => Instr::I64Ne,
            (0x53, 0) => Instr::I64LtS,
            (0x54, 0) => Instr::I64LtU,
            (0x55, 0) => Instr::I64GtS,
            (0x56, 0) => Instr::I64GtU,
            (0x57, 0) => Instr::I64LeS,
            (0x58, 0) => Instr::I64LeU,
            (0x59, 0) => Instr::I64GeS,
            (0x5a, 0) => Instr::I64GeU,
            (0x5b, 0) => Instr::F32Eq,
            (0x5c, 0) => Instr::F32Ne,
            (0x5d, 0) => Instr::F32Lt,
            (0x5e, 0) => Instr::F32Gt,
            (0x5f, 0) => Instr::F32Le,
            (0x60, 0) => Instr::F32Ge,
            (0x61, 0) => Instr::F64Eq,
            (0x62, 0) => Instr::F64Ne,
            (0x63, 0) => Instr::F64Lt,
            (0x64, 0) => Instr::F64Gt,
            (0x65, 0) => Instr::F64Le,
            (0x66, 0) => Instr::F64Ge,
            (0x67, 0) => Instr::I32Clz,
            (0x68, 0) => Instr::I32Ctz,
            (0x69, 0) => Instr::I32Popcnt,
            (0x6a, 0) => Instr::I32Add,
            (0x6b, 0) => Instr::I32Sub,
            (0x6c, 0) => Instr::I32Mul,
            (0x6d, 0) => Instr::I32DivS,
            (0x6e, 0) => Instr::I32DivU,
            (0x6f, 0) => Instr::I32RemS,
            (0x70, 0) => Instr::I32RemU,
            (0x71, 0) => Instr::I32And,
            (0x72, 0) => Instr::I32Ior,
            (0x73, 0) => Instr::I32Xor,
            (0x74, 0) => Instr::I32Shl,
            (0x75, 0) => Instr::I32ShrS,
            (0x76, 0) => Instr::I32ShrU,
            (0x77, 0) => Instr::I32Rol,
            (0x78, 0) => Instr::I32Ror,
            (0x79, 0) => Instr::I64Clz,
            (0x7a, 0) => Instr::I64Ctz,
            (0x7b, 0) => Instr::I64Popcnt,
            (0x7c, 0) => Instr::I64Add,
            (0x7d, 0) => Instr::I64Sub,
            (0x7e, 0) => Instr::I64Mul,
            (0x7f, 0) => Instr::I64DivS,
            (0x80, 0) => Instr::I64DivU,
            (0x81, 0) => Instr::I64RemS,
            (0x82, 0) => Instr::I64RemU,
            (0x83, 0) => Instr::I64And,
            (0x84, 0) => Instr::I64Ior,
            (0x85, 0) => Instr::I64Xor,
            (0x86, 0) => Instr::I64Shl,
            (0x87, 0) => Instr::I64ShrS,
            (0x88, 0) => Instr::I64ShrU,
            (0x89, 0) => Instr::I64Rol,
            (0x8a, 0) => Instr::I64Ror,
            (0x8b, 0) => Instr::F32Abs,
            (0x8c, 0) => Instr::F32Neg,
            (0x8d, 0) => Instr::F32Ceil,
            (0x8e, 0) => Instr::F32Floor,
            (0x8f, 0) => Instr::F32Trunc,
            (0x90, 0) => Instr::F32NearestInt,
            (0x91, 0) => Instr::F32Sqrt,
            (0x92, 0) => Instr::F32Add,
            (0x93, 0) => Instr::F32Sub,
            (0x94, 0) => Instr::F32Mul,
            (0x95, 0) => Instr::F32Div,
            (0x96, 0) => Instr::F32Min,
            (0x97, 0) => Instr::F32Max,
            (0x98, 0) => Instr::F32CopySign,
            (0x99, 0) => Instr::F64Abs,
            (0x9a, 0) => Instr::F64Neg,
            (0x9b, 0) => Instr::F64Ceil,
            (0x9c, 0) => Instr::F64Floor,
            (0x9d, 0) => Instr::F64Trunc,
            (0x9e, 0) => Instr::F64NearestInt,
            (0x9f, 0) => Instr::F64Sqrt,
            (0xa0, 0) => Instr::F64Add,
            (0xa1, 0) => Instr::F64Sub,
            (0xa2, 0) => Instr::F64Mul,
            (0xa3, 0) => Instr::F64Div,
            (0xa4, 0) => Instr::F64Min,
            (0xa5, 0) => Instr::F64Max,
            (0xa6, 0) => Instr::F64CopySign,
            (0xa7, 0) => Instr::I32ConvertI64,
            (0xa8, 0) => Instr::I32SConvertF32,
            (0xa9, 0) => Instr::I32UConvertF32,
            (0xaa, 0) => Instr::I32SConvertF64,
            (0xab, 0) => Instr::I32UConvertF64,
            (0xac, 0) => Instr::I64SConvertI32,
            (0xad, 0) => Instr::I64UConvertI32,
            (0xae, 0) => Instr::I64SConvertF32,
            (0xaf, 0) => Instr::I64UConvertF32,
            (0xb0, 0) => Instr::I64SConvertF64,
            (0xb1, 0) => Instr::I64UConvertF64,
            (0xb2, 0) => Instr::F32SConvertI32,
            (0xb3, 0) => Instr::F32UConvertI32,
            (0xb4, 0) => Instr::F32SConvertI64,
            (0xb5, 0) => Instr::F32UConvertI64,
            (0xb6, 0) => Instr::F32ConvertF64,
            (0xb7, 0) => Instr::F64SConvertI32,
            (0xb8, 0) => Instr::F64UConvertI32,
            (0xb9, 0) => Instr::F64SConvertI64,
            (0xba, 0) => Instr::F64UConvertI64,
            (0xbb, 0) => Instr::F64ConvertF32,
            (0xbc, 0) => Instr::I32ReinterpretF32,
            (0xbd, 0) => Instr::I64ReinterpretF64,
            (0xbe, 0) => Instr::F32ReinterpretI32,
            (0xbf, 0) => Instr::F64ReinterpretI64,
            (0xc0, 0) => Instr::I32SExtendI8,
            (0xc1, 0) => Instr::I32SExtendI16,
            (0xc2, 0) => Instr::I64SExtendI8,
            (0xc3, 0) => Instr::I64SExtendI16,
            (0xc4, 0) => Instr::I64SExtendI32,
            (0xfc, 0x00) => Instr::I32SConvertSatF32,
            (0xfc, 0x01) => Instr::I32UConvertSatF32,
            (0xfc, 0x02) => Instr::I32SConvertSatF64,
            (0xfc, 0x03) => Instr::I32UConvertSatF64,
            (0xfc, 0x04) => Instr::I64SConvertSatF32,
            (0xfc, 0x05) => Instr::I64UConvertSatF32,
            (0xfc, 0x06) => Instr::I64SConvertSatF64,
            (0xfc, 0x07) => Instr::I64UConvertSatF64,
            _ => return Err(DefaultIRGeneratorError::InvalidInstruction(code, subcode)),
        })?);
        Ok(())
    }

    fn make_instr_block(
        &mut self,
        block_kind: u8,
        block_type: Self::BlockType,
        expr: Self::Expr,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        instrs.push(self.trace(match block_kind {
            0x02 => Instr::Block(block_type, expr.0.into_boxed_slice()),
            0x03 => Instr::Loop(block_type, expr.0.into_boxed_slice()),
            unk => return Err(DefaultIRGeneratorError::InvalidInstruction(unk, 0)),
        })?);
        Ok(())
    }

    fn make_instr_block_ifelse(
        &mut self,
        block_type: Self::BlockType,
        consequent: Self::Expr,
        alternate: Option<Self::Expr>,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        instrs.push(self.trace(if let Some(alternate) = alternate {
            Instr::IfElse(
                block_type,
                consequent.0.into_boxed_slice(),
                alternate.0.into_boxed_slice(),
            )
        } else {
            Instr::If(block_type, consequent.0.into_boxed_slice())
        })?);
        Ok(())
    }

    fn make_elem_from_indices(
        &mut self,
        kind: Option<u32>,
        mode: Self::ElemMode,
        idxs: Box<[u32]>,
        flags: u8,
    ) -> Result<Self::Elem, Self::Error> {
        // TODO: actually check the types...
        self.type_checker.clear();
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
        self.type_checker.clear();
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

impl DefaultIRGenerator {
    fn trace(&mut self, instr: Instr) -> Result<Instr, DefaultIRGeneratorError> {
        self.type_checker
            .trace(
                instr,
                self.func_types
                    .as_ref()
                    .map(|xs| xs as &[_])
                    .unwrap_or_default(),
                self.types.as_ref().map(|xs| xs as &[_]).unwrap_or_default(),
                &self.current_locals,
                &self.global_types,
                self.table_types
                    .as_ref()
                    .map(|xs| xs as &[_])
                    .unwrap_or_default(),
                self.global_import_boundary_idx,
            )
            .map_err(Into::into)
    }

    fn types(&self) -> &[Type] {
        self.types.as_ref().map(|xs| xs as &[_]).unwrap_or_default()
    }

    fn func_types(&self) -> &[TypeIdx] {
        self.func_types
            .as_ref()
            .map(|xs| xs as &[_])
            .unwrap_or_default()
    }

    fn block_type(&self, block_type: &BlockType) -> (Box<[ValType]>, Box<[ValType]>) {
        match block_type {
            BlockType::Empty => (Box::new([]), Box::new([])),
            BlockType::Val(v) => (Box::new([]), Box::new([*v])),
            BlockType::TypeIndex(TypeIdx(idx)) => {
                let typedef = &self.types()[*idx as usize];
                (typedef.0 .0.clone(), typedef.1 .0.clone())
            }
        }
    }

    fn func_type(&self, FuncIdx(func_idx): &FuncIdx) -> (Box<[ValType]>, Box<[ValType]>) {
        let TypeIdx(idx) = &self.func_types()[*func_idx as usize];
        let typedef = &self.types()[*idx as usize];
        (typedef.0 .0.clone(), typedef.1 .0.clone())
    }
}
