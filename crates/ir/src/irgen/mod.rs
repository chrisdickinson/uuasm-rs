mod default;
mod empty;
pub use default::*;
pub use empty::*;
use std::error::Error;

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

    const IS_MULTIMEMORY_ENABLED: bool = false;

    fn make_instr_select(
        &mut self,
        types: Box<[Self::ValType]>,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error>;

    fn make_instr_table(
        &mut self,
        items: &[u32],
        alternate: u32,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error>;

    fn make_instr_arity1_64(
        &mut self,
        code: u8,
        subcode: u32,
        arg0: u64,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error>;

    fn make_instr_arity2(
        &mut self,
        code: u8,
        subcode: u32,
        arg0: u32,
        arg1: u32,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error>;

    fn make_instr_arity1(
        &mut self,
        code: u8,
        subcode: u32,
        arg0: u32,
        instrs: &mut Vec<Self::Instr>,
    ) -> Result<(), Self::Error>;

    fn make_instr_arity0(
        &mut self,
        code: u8,
        subcode: u32,
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

    fn start_elem_expr(&mut self, _kind: Option<&Self::RefType>) -> Result<(), Self::Error> {
        Ok(())
    }

    fn check_elem_expr(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

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

    fn start_data_offset(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn start_global(&mut self, _global_type: &Self::GlobalType) -> Result<(), Self::Error> {
        Ok(())
    }
    fn start_elem_reftype_list(
        &mut self,
        _ref_type: Option<&Self::RefType>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
    fn start_elem_active_table_index(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn start_section(&mut self, _section_id: u8, _section_size: u32) -> Result<(), Self::Error> {
        Ok(())
    }
    fn start_block(&mut self, _block_type: &Self::BlockType) -> Result<(), Self::Error> {
        Ok(())
    }
    fn start_loop(&mut self, _block_type: &Self::BlockType) -> Result<(), Self::Error> {
        Ok(())
    }
    fn start_if(&mut self, _block_type: &Self::BlockType) -> Result<(), Self::Error> {
        Ok(())
    }
    fn start_else(&mut self, _block_type: &Self::BlockType) -> Result<(), Self::Error> {
        Ok(())
    }
    fn start_func(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

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

    fn make_custom_section(
        &mut self,
        name: String,
        payload: Box<[u8]>,
    ) -> Result<Self::Section, Self::Error>;

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
    fn make_data_index(&mut self, candidate: u32) -> Result<Self::DataIdx, Self::Error>;
    fn make_elem_index(&mut self, candidate: u32) -> Result<Self::ElemIdx, Self::Error>;
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
