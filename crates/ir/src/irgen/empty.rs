use super::IR;
use std::{error::Error, fmt::Debug};

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
        _types: Box<[Self::ValType]>,
        _instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn make_instr_table(
        &mut self,
        _items: &[u32],
        _alternate: u32,
        _instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn make_instr_arity1_64(
        &mut self,
        _code: u8,
        _subcode: u32,
        _arg0: u64,
        _instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn make_instr_arity2(
        &mut self,
        _code: u8,
        _subcode: u32,
        _arg0: u32,
        _arg1: u32,
        _instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn make_instr_arity1(
        &mut self,
        _code: u8,
        _subcode: u32,
        _arg0: u32,
        _instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn make_instr_arity0(
        &mut self,
        _code: u8,
        _subcode: u32,
        _instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn make_instr_block(
        &mut self,
        _block_kind: u8,
        _block_type: Self::BlockType,
        _expr: Self::Expr,
        _instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn make_instr_block_ifelse(
        &mut self,
        _block_type: Self::BlockType,
        _consequent: Self::Expr,
        _alternate: Option<Self::Expr>,
        _instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn make_code(&mut self, _item: Self::Func) -> Result<Self::Code, Self::Error> {
        Ok(())
    }

    fn make_data_active(
        &mut self,
        _bytes: Box<[u8]>,
        _mem_idx: Self::MemIdx,
        _expr: Self::Expr,
    ) -> Result<Self::Data, Self::Error> {
        Ok(())
    }
    fn make_data_passive(&mut self, _bytes: Box<[u8]>) -> Result<Self::Data, Self::Error> {
        Ok(())
    }

    fn make_limits(
        &mut self,
        _lower: u32,
        _upper: Option<u32>,
    ) -> Result<Self::Limits, Self::Error> {
        Ok(())
    }

    fn make_export(
        &mut self,
        _name: Self::Name,
        _desc: Self::ExportDesc,
    ) -> Result<Self::Export, Self::Error> {
        Ok(())
    }

    fn make_expr(&mut self, _instrs: Vec<Self::Instr>) -> Result<Self::Expr, Self::Error> {
        Ok(())
    }

    fn make_func(
        &mut self,
        _locals: Box<[Self::Local]>,
        _expr: Self::Expr,
    ) -> Result<Self::Func, Self::Error> {
        Ok(())
    }

    fn make_global(
        &mut self,
        _global_type: Self::GlobalType,
        _expr: Self::Expr,
    ) -> Result<Self::Global, Self::Error> {
        Ok(())
    }

    fn make_local(
        &mut self,
        _count: u32,
        _val_type: Self::ValType,
    ) -> Result<Self::Local, Self::Error> {
        Ok(())
    }

    fn make_name(&mut self, _data: Box<[u8]>) -> Result<Self::Name, Self::Error> {
        Ok(())
    }
    fn make_custom_section(&mut self, _data: Box<[u8]>) -> Result<Self::Section, Self::Error> {
        Ok(())
    }
    fn make_type_section(
        &mut self,
        _data: Box<[Self::Type]>,
    ) -> Result<Self::Section, Self::Error> {
        Ok(())
    }
    fn make_import_section(
        &mut self,
        _data: Box<[Self::Import]>,
    ) -> Result<Self::Section, Self::Error> {
        Ok(())
    }
    fn make_function_section(
        &mut self,
        _data: Box<[Self::TypeIdx]>,
    ) -> Result<Self::Section, Self::Error> {
        Ok(())
    }
    fn make_table_section(
        &mut self,
        _data: Box<[Self::TableType]>,
    ) -> Result<Self::Section, Self::Error> {
        Ok(())
    }
    fn make_memory_section(
        &mut self,
        _data: Box<[Self::MemType]>,
    ) -> Result<Self::Section, Self::Error> {
        Ok(())
    }
    fn make_global_section(
        &mut self,
        _data: Box<[Self::Global]>,
    ) -> Result<Self::Section, Self::Error> {
        Ok(())
    }
    fn make_export_section(
        &mut self,
        _data: Box<[Self::Export]>,
    ) -> Result<Self::Section, Self::Error> {
        Ok(())
    }

    fn make_start_section(&mut self, _data: Self::FuncIdx) -> Result<Self::Section, Self::Error> {
        Ok(())
    }

    fn make_element_section(
        &mut self,
        _data: Box<[Self::Elem]>,
    ) -> Result<Self::Section, Self::Error> {
        Ok(())
    }

    fn make_code_section(
        &mut self,
        _data: Box<[Self::Code]>,
    ) -> Result<Self::Section, Self::Error> {
        Ok(())
    }
    fn make_data_section(
        &mut self,
        _data: Box<[Self::Data]>,
    ) -> Result<Self::Section, Self::Error> {
        Ok(())
    }

    fn make_datacount_section(&mut self, _data: u32) -> Result<Self::Section, Self::Error> {
        Ok(())
    }

    fn make_block_type_empty(&mut self) -> Result<Self::BlockType, Self::Error> {
        Ok(())
    }
    fn make_block_type_val_type(
        &mut self,
        _vt: Self::ValType,
    ) -> Result<Self::BlockType, Self::Error> {
        Ok(())
    }
    fn make_block_type_type_index(
        &mut self,
        _ti: Self::TypeIdx,
    ) -> Result<Self::BlockType, Self::Error> {
        Ok(())
    }

    fn make_val_type(&mut self, _data: u8) -> Result<Self::ValType, Self::Error> {
        Ok(())
    }
    fn make_global_type(
        &mut self,
        _valtype: Self::ValType,
        _is_mutable: bool,
    ) -> Result<Self::GlobalType, Self::Error> {
        Ok(())
    }
    fn make_table_type(
        &mut self,
        _reftype_candidate: u8,
        _limits: Self::Limits,
    ) -> Result<Self::TableType, Self::Error> {
        Ok(())
    }
    fn make_mem_type(&mut self, _limits: Self::Limits) -> Result<Self::MemType, Self::Error> {
        Ok(())
    }

    fn make_result_type(&mut self, _data: &[u8]) -> Result<Self::ResultType, Self::Error> {
        Ok(())
    }
    fn make_type_index(&mut self, _candidate: u32) -> Result<Self::TypeIdx, Self::Error> {
        Ok(())
    }
    fn make_table_index(&mut self, _candidate: u32) -> Result<Self::TableIdx, Self::Error> {
        Ok(())
    }
    fn make_mem_index(&mut self, _candidate: u32) -> Result<Self::MemIdx, Self::Error> {
        Ok(())
    }
    fn make_global_index(&mut self, _candidate: u32) -> Result<Self::GlobalIdx, Self::Error> {
        Ok(())
    }
    fn make_func_index(&mut self, _candidate: u32) -> Result<Self::FuncIdx, Self::Error> {
        Ok(())
    }
    fn make_local_index(&mut self, _candidate: u32) -> Result<Self::FuncIdx, Self::Error> {
        Ok(())
    }
    fn make_data_index(&mut self, _candidate: u32) -> Result<Self::DataIdx, Self::Error> {
        Ok(())
    }
    fn make_elem_index(&mut self, _candidate: u32) -> Result<Self::ElemIdx, Self::Error> {
        Ok(())
    }
    fn make_func_type(
        &mut self,
        _params: Option<Self::ResultType>,
        _returns: Option<Self::ResultType>,
    ) -> Result<Self::Type, Self::Error> {
        Ok(())
    }

    fn make_import_desc_func(
        &mut self,
        _type_idx: Self::TypeIdx,
    ) -> Result<Self::ImportDesc, Self::Error> {
        Ok(())
    }
    fn make_import_desc_global(
        &mut self,
        _global_type: Self::GlobalType,
    ) -> Result<Self::ImportDesc, Self::Error> {
        Ok(())
    }
    fn make_import_desc_table(
        &mut self,
        _global_type: Self::TableType,
    ) -> Result<Self::ImportDesc, Self::Error> {
        Ok(())
    }
    fn make_import_desc_memtype(
        &mut self,
        _global_type: Self::Limits,
    ) -> Result<Self::ImportDesc, Self::Error> {
        Ok(())
    }

    fn make_export_desc_func(
        &mut self,
        _func_idx: Self::FuncIdx,
    ) -> Result<Self::ExportDesc, Self::Error> {
        Ok(())
    }

    fn make_export_desc_global(
        &mut self,
        _global_idx: Self::GlobalIdx,
    ) -> Result<Self::ExportDesc, Self::Error> {
        Ok(())
    }

    fn make_export_desc_memtype(
        &mut self,
        _mem_idx: Self::MemIdx,
    ) -> Result<Self::ExportDesc, Self::Error> {
        Ok(())
    }

    fn make_export_desc_table(
        &mut self,
        _table_idx: Self::TableIdx,
    ) -> Result<Self::ExportDesc, Self::Error> {
        Ok(())
    }

    fn make_import(
        &mut self,
        _modname: Self::Name,
        _name: Self::Name,
        _desc: Self::ImportDesc,
    ) -> Result<Self::Import, Self::Error> {
        Ok(())
    }
    fn make_module(&mut self, _sections: Vec<Self::Section>) -> Result<Self::Module, Self::Error> {
        Ok(())
    }

    fn make_elem_from_indices(
        &mut self,
        __kind: Option<u32>,
        __mode: Self::ElemMode,
        __idxs: Box<[u32]>,
        __flags: u8,
    ) -> Result<Self::Elem, Self::Error> {
        Ok(())
    }

    fn make_elem_from_exprs(
        &mut self,
        __kind: Option<Self::RefType>,
        __mode: Self::ElemMode,
        __exprs: Box<[Self::Expr]>,
        __flags: u8,
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
        _table_idx: Self::TableIdx,
        _expr: Self::Expr,
    ) -> Result<Self::ElemMode, Self::Error> {
        Ok(())
    }

    fn make_ref_type(&mut self, _data: u8) -> Result<Self::RefType, Self::Error> {
        Ok(())
    }
}
