#![allow(dead_code)]
#![allow(clippy::upper_case_acronyms)]

use std::collections::HashMap;

use anyhow::Context;

use crate::{
    memory_region::MemoryRegion,
    nodes::{
        BlockType, ByteVec, CodeIdx, Data, Elem, ExportDesc, Expr, FuncIdx, Global, GlobalType,
        ImportDesc, Instr, MemType, Module as ParsedModule, NumType, RefType,
        TableType, TypeIdx, ValType, VecType, GlobalIdx, Mutability,
    },
};

#[derive(Debug)]
struct TKTK;

macro_rules! Instrs {
    ($input:expr, $exec:ident) => {
        match $input {
            Instr::Unreachable => $exec!(Instr::Unreachable),
            Instr::Nop => $exec!(Instr::Nop),
            Instr::Block(a, b) => $exec!(Instr::Block, a, b)
            Instr::Loop(a, b) => $exec!(Instr::Loop, a, b)
            Instr::If(a, b) => $exec!(Instr::If, a, b)
            Instr::IfElse(a, b, c) => $exec!(Instr::IfElse, a, b, c)
            Instr::Br(a) => $exec!(Instr::Br, a)
            Instr::BrIf(a) => $exec!(Instr::BrIf, a)
            Instr::BrTable(a, b) => $exec!(Instr::BrTable, a, b)
            Instr::Return => $exec!(Instr::Return),
            Instr::Call(a) => $exec!(Instr::Call, a)
            Instr::CallIndirect(a, b) => $exec!(Instr::CallIndirect, a, b)
            Instr::RefNull(a) => $exec!(Instr::RefNull, a)
            Instr::RefIsNull => $exec!(Instr::RefIsNull),
            Instr::RefFunc(a) => $exec!(Instr::RefFunc, a)
            Instr::Drop => $exec!(Instr::Drop),
            Instr::SelectEmpty => $exec!(Instr::SelectEmpty),
            Instr::Select(a) => $exec!(Instr::Select, a)
            Instr::LocalGet(a) => $exec!(Instr::LocalGet, a)
            Instr::LocalSet(a) => $exec!(Instr::LocalSet, a)
            Instr::LocalTee(a) => $exec!(Instr::LocalTee, a)
            Instr::GlobalGet(a) => $exec!(Instr::GlobalGet, a)
            Instr::GlobalSet(a) => $exec!(Instr::GlobalSet, a)
            Instr::TableGet(a) => $exec!(Instr::TableGet, a)
            Instr::TableSet(a) => $exec!(Instr::TableSet, a)
            Instr::TableInit(a, b) => $exec!(Instr::TableInit, a, b)
            Instr::ElemDrop(a) => $exec!(Instr::ElemDrop, a)
            Instr::TableCopy(a, b) => $exec!(Instr::TableCopy, a, b)
            Instr::TableGrow(a) => $exec!(Instr::TableGrow, a)
            Instr::TableSize(a) => $exec!(Instr::TableSize, a)
            Instr::TableFill(a) => $exec!(Instr::TableFill, a)
            Instr::I32Load(mem) => $exec!(Instr::I32Load, mem),
            Instr::I64Load(mem) => $exec!(Instr::I64Load, mem),
            Instr::F32Load(mem) => $exec!(Instr::F32Load, mem),
            Instr::F64Load(mem) => $exec!(Instr::F64Load, mem),
            Instr::I32Load8S(mem) => $exec!(Instr::I32Load8S, mem),
            Instr::I32Load8U(mem) => $exec!(Instr::I32Load8U, mem),
            Instr::I32Load16S(mem) => $exec!(Instr::I32Load16S, mem),
            Instr::I32Load16U(mem) => $exec!(Instr::I32Load16U, mem),
            Instr::I64Load8S(mem) => $exec!(Instr::I64Load8S, mem),
            Instr::I64Load8U(mem) => $exec!(Instr::I64Load8U, mem),
            Instr::I64Load16S(mem) => $exec!(Instr::I64Load16S, mem),
            Instr::I64Load16U(mem) => $exec!(Instr::I64Load16U, mem),
            Instr::I64Load32S(mem) => $exec!(Instr::I64Load32S, mem),
            Instr::I64Load32U(mem) => $exec!(Instr::I64Load32U, mem),
            Instr::I32Store(a) => $exec!(Instr::I32Store, a)
            Instr::I64Store(a) => $exec!(Instr::I64Store, a)
            Instr::F32Store(a) => $exec!(Instr::F32Store, a)
            Instr::F64Store(a) => $exec!(Instr::F64Store, a)
            Instr::I32Store8(a) => $exec!(Instr::I32Store8, a)
            Instr::I32Store16(a) => $exec!(Instr::I32Store16, a)
            Instr::I64Store8(a) => $exec!(Instr::I64Store8, a)
            Instr::I64Store16(a) => $exec!(Instr::I64Store16, a)
            Instr::I64Store32(a) => $exec!(Instr::I64Store32, a)
            Instr::MemorySize(a) => $exec!(Instr::MemorySize, a)
            Instr::MemoryGrow(a) => $exec!(Instr::MemoryGrow, a)
            Instr::MemoryInit(a, b) => $exec!(Instr::MemoryInit, a, b)
            Instr::DataDrop(a) => $exec!(Instr::DataDrop, a)
            Instr::MemoryCopy(a, b) => $exec!(Instr::MemoryCopy, a, b)
            Instr::MemoryFill(a) => $exec!(Instr::MemoryFill, a)
            Instr::I32Const(a) => $exec!(Instr::I32Const, a)
            Instr::I64Const(a) => $exec!(Instr::I64Const, a)
            Instr::F32Const(a) => $exec!(Instr::F32Const, a)
            Instr::F64Const(a) => $exec!(Instr::F64Const, a)
            Instr::I32Eqz => $exec!(Instr::I32Eqz),
            Instr::I32Eq => $exec!(Instr::I32Eq),
            Instr::I32Ne => $exec!(Instr::I32Ne),
            Instr::I32LtS => $exec!(Instr::I32LtS),
            Instr::I32LtU => $exec!(Instr::I32LtU),
            Instr::I32GtS => $exec!(Instr::I32GtS),
            Instr::I32GtU => $exec!(Instr::I32GtU),
            Instr::I32LeS => $exec!(Instr::I32LeS),
            Instr::I32LeU => $exec!(Instr::I32LeU),
            Instr::I32GeS => $exec!(Instr::I32GeS),
            Instr::I32GeU => $exec!(Instr::I32GeU),
            Instr::I64Eqz => $exec!(Instr::I64Eqz),
            Instr::I64Eq => $exec!(Instr::I64Eq),
            Instr::I64Ne => $exec!(Instr::I64Ne),
            Instr::I64LtS => $exec!(Instr::I64LtS),
            Instr::I64LtU => $exec!(Instr::I64LtU),
            Instr::I64GtS => $exec!(Instr::I64GtS),
            Instr::I64GtU => $exec!(Instr::I64GtU),
            Instr::I64LeS => $exec!(Instr::I64LeS),
            Instr::I64LeU => $exec!(Instr::I64LeU),
            Instr::I64GeS => $exec!(Instr::I64GeS),
            Instr::I64GeU => $exec!(Instr::I64GeU),
            Instr::F32Eq => $exec!(Instr::F32Eq),
            Instr::F32Ne => $exec!(Instr::F32Ne),
            Instr::F32Lt => $exec!(Instr::F32Lt),
            Instr::F32Gt => $exec!(Instr::F32Gt),
            Instr::F32Le => $exec!(Instr::F32Le),
            Instr::F32Ge => $exec!(Instr::F32Ge),
            Instr::F64Eq => $exec!(Instr::F64Eq),
            Instr::F64Ne => $exec!(Instr::F64Ne),
            Instr::F64Lt => $exec!(Instr::F64Lt),
            Instr::F64Gt => $exec!(Instr::F64Gt),
            Instr::F64Le => $exec!(Instr::F64Le),
            Instr::F64Ge => $exec!(Instr::F64Ge),
            Instr::I32Clz => $exec!(Instr::I32Clz),
            Instr::I32Ctz => $exec!(Instr::I32Ctz),
            Instr::I32Popcnt => $exec!(Instr::I32Popcnt),
            Instr::I32Add => $exec!(Instr::I32Add),
            Instr::I32Sub => $exec!(Instr::I32Sub),
            Instr::I32Mul => $exec!(Instr::I32Mul),
            Instr::I32DivS => $exec!(Instr::I32DivS),
            Instr::I32DivU => $exec!(Instr::I32DivU),
            Instr::I32RemS => $exec!(Instr::I32RemS),
            Instr::I32RemU => $exec!(Instr::I32RemU),
            Instr::I32And => $exec!(Instr::I32And),
            Instr::I32Ior => $exec!(Instr::I32Ior),
            Instr::I32Xor => $exec!(Instr::I32Xor),
            Instr::I32Shl => $exec!(Instr::I32Shl),
            Instr::I32ShrS => $exec!(Instr::I32ShrS),
            Instr::I32ShrU => $exec!(Instr::I32ShrU),
            Instr::I32Rol => $exec!(Instr::I32Rol),
            Instr::I32Ror => $exec!(Instr::I32Ror),
            Instr::I64Clz => $exec!(Instr::I64Clz),
            Instr::I64Ctz => $exec!(Instr::I64Ctz),
            Instr::I64Popcnt => $exec!(Instr::I64Popcnt),
            Instr::I64Add => $exec!(Instr::I64Add),
            Instr::I64Sub => $exec!(Instr::I64Sub),
            Instr::I64Mul => $exec!(Instr::I64Mul),
            Instr::I64DivS => $exec!(Instr::I64DivS),
            Instr::I64DivU => $exec!(Instr::I64DivU),
            Instr::I64RemS => $exec!(Instr::I64RemS),
            Instr::I64RemU => $exec!(Instr::I64RemU),
            Instr::I64And => $exec!(Instr::I64And),
            Instr::I64Ior => $exec!(Instr::I64Ior),
            Instr::I64Xor => $exec!(Instr::I64Xor),
            Instr::I64Shl => $exec!(Instr::I64Shl),
            Instr::I64ShrS => $exec!(Instr::I64ShrS),
            Instr::I64ShrU => $exec!(Instr::I64ShrU),
            Instr::I64Rol => $exec!(Instr::I64Rol),
            Instr::I64Ror => $exec!(Instr::I64Ror),
            Instr::F32Abs => $exec!(Instr::F32Abs),
            Instr::F32Neg => $exec!(Instr::F32Neg),
            Instr::F32Ceil => $exec!(Instr::F32Ceil),
            Instr::F32Floor => $exec!(Instr::F32Floor),
            Instr::F32Trunc => $exec!(Instr::F32Trunc),
            Instr::F32NearestInt => $exec!(Instr::F32NearestInt),
            Instr::F32Sqrt => $exec!(Instr::F32Sqrt),
            Instr::F32Add => $exec!(Instr::F32Add),
            Instr::F32Sub => $exec!(Instr::F32Sub),
            Instr::F32Mul => $exec!(Instr::F32Mul),
            Instr::F32Div => $exec!(Instr::F32Div),
            Instr::F32Min => $exec!(Instr::F32Min),
            Instr::F32Max => $exec!(Instr::F32Max),
            Instr::F32CopySign => $exec!(Instr::F32CopySign),
            Instr::F64Abs => $exec!(Instr::F64Abs),
            Instr::F64Neg => $exec!(Instr::F64Neg),
            Instr::F64Ceil => $exec!(Instr::F64Ceil),
            Instr::F64Floor => $exec!(Instr::F64Floor),
            Instr::F64Trunc => $exec!(Instr::F64Trunc),
            Instr::F64NearestInt => $exec!(Instr::F64NearestInt),
            Instr::F64Sqrt => $exec!(Instr::F64Sqrt),
            Instr::F64Add => $exec!(Instr::F64Add),
            Instr::F64Sub => $exec!(Instr::F64Sub),
            Instr::F64Mul => $exec!(Instr::F64Mul),
            Instr::F64Div => $exec!(Instr::F64Div),
            Instr::F64Min => $exec!(Instr::F64Min),
            Instr::F64Max => $exec!(Instr::F64Max),
            Instr::F64CopySign => $exec!(Instr::F64CopySign),
            Instr::I32ConvertI64 => $exec!(Instr::I32ConvertI64),
            Instr::I32SConvertF32 => $exec!(Instr::I32SConvertF32),
            Instr::I32UConvertF32 => $exec!(Instr::I32UConvertF32),
            Instr::I32SConvertF64 => $exec!(Instr::I32SConvertF64),
            Instr::I32UConvertF64 => $exec!(Instr::I32UConvertF64),
            Instr::I64SConvertI32 => $exec!(Instr::I64SConvertI32),
            Instr::I64UConvertI32 => $exec!(Instr::I64UConvertI32),
            Instr::I64SConvertF32 => $exec!(Instr::I64SConvertF32),
            Instr::I64UConvertF32 => $exec!(Instr::I64UConvertF32),
            Instr::I64SConvertF64 => $exec!(Instr::I64SConvertF64),
            Instr::I64UConvertF64 => $exec!(Instr::I64UConvertF64),
            Instr::F32SConvertI32 => $exec!(Instr::F32SConvertI32),
            Instr::F32UConvertI32 => $exec!(Instr::F32UConvertI32),
            Instr::F32SConvertI64 => $exec!(Instr::F32SConvertI64),
            Instr::F32UConvertI64 => $exec!(Instr::F32UConvertI64),
            Instr::F32ConvertF64 => $exec!(Instr::F32ConvertF64),
            Instr::F64SConvertI32 => $exec!(Instr::F64SConvertI32),
            Instr::F64UConvertI32 => $exec!(Instr::F64UConvertI32),
            Instr::F64SConvertI64 => $exec!(Instr::F64SConvertI64),
            Instr::F64UConvertI64 => $exec!(Instr::F64UConvertI64),
            Instr::F64ConvertF32 => $exec!(Instr::F64ConvertF32),
            Instr::I32ReinterpretF32 => $exec!(Instr::I32ReinterpretF32),
            Instr::I64ReinterpretF64 => $exec!(Instr::I64ReinterpretF64),
            Instr::F32ReinterpretI32 => $exec!(Instr::F32ReinterpretI32),
            Instr::F64ReinterpretI64 => $exec!(Instr::F64ReinterpretI64),
            Instr::I32SExtendI8 => $exec!(Instr::I32SExtendI8),
            Instr::I32SExtendI16 => $exec!(Instr::I32SExtendI16),
            Instr::I64SExtendI8 => $exec!(Instr::I64SExtendI8),
            Instr::I64SExtendI16 => $exec!(Instr::I64SExtendI16),
            Instr::I64SExtendI32 => $exec!(Instr::I64SExtendI32),
        }
    };
}


impl ValType {
    fn instantiate(&self) -> Value {
        match self {
            ValType::NumType(NumType::I32) => Value::I32(Default::default()),
            ValType::NumType(NumType::I64) => Value::I64(Default::default()),
            ValType::NumType(NumType::F32) => Value::F32(Default::default()),
            ValType::NumType(NumType::F64) => Value::F64(Default::default()),
            ValType::VecType(VecType::V128) => Value::V128(Default::default()),
            ValType::RefType(RefType::FuncRef) => Value::RefNull,
            ValType::RefType(RefType::ExternRef) => Value::RefNull,
        }
    }

    #[inline]
    fn validate(&self, value: &Value) -> anyhow::Result<()> {
        match (self, value) {
            (ValType::NumType(NumType::I32), Value::I32(_))
            | (ValType::NumType(NumType::I64), Value::I64(_))
            | (ValType::NumType(NumType::F32), Value::F32(_))
            | (ValType::NumType(NumType::F64), Value::F64(_))
            | (ValType::VecType(VecType::V128), Value::V128(_))
            | (ValType::RefType(RefType::FuncRef), Value::RefFunc(_))
            | (ValType::RefType(RefType::FuncRef), Value::RefNull)
            | (ValType::RefType(RefType::ExternRef), Value::RefExtern(_))
            | (ValType::RefType(RefType::ExternRef), Value::RefNull) => {
                Ok(())
            }
            (vt, v) => anyhow::bail!(
                "expected={:?}; got={:?}",
                vt,
                v
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Value {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    V128(i128),
    RefNull,
    RefFunc(FuncIdx),
    RefExtern(u32),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::I32(l0), Self::I32(r0)) => l0 == r0,
            (Self::I64(l0), Self::I64(r0)) => l0 == r0,
            (Self::F32(l0), Self::F32(r0)) => l0 == r0,
            (Self::F64(l0), Self::F64(r0)) => l0 == r0,
            (Self::V128(l0), Self::V128(r0)) => l0 == r0,
            (Self::RefFunc(l0), Self::RefFunc(r0)) => l0 == r0,
            (Self::RefExtern(l0), Self::RefExtern(r0)) => l0 == r0,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl Value {
    #[cfg(test)]
    pub(crate) fn bit_eq(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::F32(l0), Self::F32(r0)) => unsafe {
                std::mem::transmute::<&f32, &u32>(l0) == std::mem::transmute::<&f32, &u32>(r0)
            },
            (Self::F64(l0), Self::F64(r0)) => unsafe {
                std::mem::transmute::<&f64, &u64>(l0) == std::mem::transmute::<&f64, &u64>(r0)
            },
            (lhs, rhs) => lhs == rhs,
        }
    }

    fn as_usize(&self) -> Option<usize> {
        Some(match self {
            Value::I32(xs) => *xs as usize,
            Value::I64(xs) => *xs as usize,
            _ => return None,
        })
    }

    // This is distinct from "as_usize": i64's are not valid candidates
    // for memory offsets until mem64 lands in wasm.
    fn as_mem_offset(&self) -> anyhow::Result<usize> {
        self.as_i32().map(|xs| xs as usize)
    }

    fn as_i32(&self) -> anyhow::Result<i32> {
        Ok(match self {
            Value::I32(xs) => *xs,
            _ => anyhow::bail!("expected i32 value"),
        })
    }

    fn is_zero(&self) -> anyhow::Result<bool> {
        Ok(match self {
            Value::I32(xs) => *xs == 0,
            Value::I64(xs) => *xs == 0,
            Value::F32(xs) => *xs == 0.0,
            Value::F64(xs) => *xs == 0.0,
            _ => anyhow::bail!("invalid conversion of value to int via is_zero"),
        })
    }
}

#[derive(Debug)]
enum FuncInstImpl {
    Guest(CodeIdx),
    Host(TKTK),
}

#[derive(Debug)]
struct FuncInst {
    r#type: TypeIdx,
    r#impl: FuncInstImpl,
}

#[derive(Debug)]
enum TableInstImpl {
    Guest(Vec<Value>),
    Host(TKTK),
}

#[derive(Debug)]
struct TableInst {
    r#type: TableType,
    r#impl: TableInstImpl,
}

#[derive(Debug)]
enum MemoryInstImpl {
    Guest(MemoryRegion),
    Host(TKTK),
}

#[derive(Debug)]
struct MemInst {
    r#type: MemType,
    r#impl: MemoryInstImpl,
}

impl MemInst {
    fn copy_data(&mut self, byte_vec: &ByteVec<'_>, offset: usize) -> anyhow::Result<()> {
        match &mut self.r#impl {
            MemoryInstImpl::Guest(region) => region.copy_data(byte_vec.0, offset)?,
            MemoryInstImpl::Host(_tktk) => todo!(),
        }

        Ok(())
    }

    #[inline]
    fn grow(&mut self, page_count: usize) -> anyhow::Result<usize> {
        match &mut self.r#impl {
            MemoryInstImpl::Guest(memory) => {
                memory.grow(page_count)
            }
            MemoryInstImpl::Host(_) => todo!(),
        }
    }

    #[inline]
    fn load<const U: usize>(&self, addr: usize) -> anyhow::Result<[u8; U]> {
        match &self.r#impl {
            MemoryInstImpl::Guest(memory) => {
                if addr.saturating_add(U) > memory.len() {
                    anyhow::bail!("out of bounds memory access")
                }
                Ok(memory.as_slice()[addr..addr + U].try_into()?)
            }
            MemoryInstImpl::Host(_) => todo!(),
        }
    }

    #[inline]
    fn write<const U: usize>(&mut self, addr: usize, value: &[u8; U]) -> anyhow::Result<()> {
        match &self.r#impl {
            MemoryInstImpl::Guest(memory) => {
                if addr.saturating_add(U) > memory.len() {
                    anyhow::bail!("out of bounds memory access")
                }
                memory.write(addr, value);

                Ok(())
            }
            MemoryInstImpl::Host(_) => todo!(),
        }
    }
}

#[derive(Debug)]
enum GlobalInstImpl {
    Guest(Value),
    Host(TKTK),
}

#[derive(Debug)]
struct GlobalInst {
    r#type: GlobalType,
    r#impl: GlobalInstImpl,
}

impl GlobalInst {
    fn value(&self) -> Value {
        match self.r#impl {
            GlobalInstImpl::Guest(v) => v,
            GlobalInstImpl::Host(_) => todo!(),
        }
    }
}

type ExportInst = TKTK;

// This should really be an ECS store.
#[derive(Debug)]
pub(crate) struct Module<'a> {
    parsed_module: ParsedModule<'a>,
}

#[derive(Debug)]
pub(crate) struct ModuleInstance<'a> {
    module: &'a Module<'a>,
    exports: HashMap<&'a str, ExportDesc>,
    functions: Vec<FuncInst>,
    globals: Vec<GlobalInst>,
    memories: Vec<MemInst>,
    tables: Vec<TableInst>,
}

enum Extern {
    Func(TKTK),
    Global(TKTK),
    Table(TKTK),
    Memory(TKTK),
    SharedMemory(TKTK),
}

pub(crate) struct Imports {
    globals: Vec<Value>,
}

impl Imports {
    pub(crate) fn new(globals: Vec<Value>) -> Self {
        Self { globals }
    }
}

impl<'a> Module<'a> {
    pub(crate) fn new<'b: 'a>(module: ParsedModule<'b>) -> Self {
        Self {
            parsed_module: module,
        }
    }

    pub(crate) fn instantiate(&self, _imports: &Imports) -> anyhow::Result<ModuleInstance> {
        let (func_imports, memory_imports, table_imports, global_imports) = self
            .parsed_module
            .import_section()
            .iter()
            .flat_map(|xs| xs.iter())
            .fold(
                (vec![], vec![], vec![], vec![]),
                |(mut funcs, mut mems, mut tables, mut globals), imp| {
                    match imp.desc {
                        ImportDesc::Func(desc) => funcs.push(desc),
                        ImportDesc::Mem(desc) => mems.push(desc),
                        ImportDesc::Table(desc) => tables.push(desc),
                        ImportDesc::Global(desc) => globals.push(desc),
                    }

                    (funcs, mems, tables, globals)
                },
            );

        let globals: Vec<GlobalInst> = global_imports
            .into_iter()
            .map(|desc| GlobalInst {
                r#type: desc,
                r#impl: GlobalInstImpl::Host(TKTK),
            })
            .collect();

        let globals = self
            .parsed_module
            .global_section()
            .iter()
            .flat_map(|xs| xs.iter())
            .try_fold(globals, |mut globals, global| {
                let Global(global_type, Expr(instrs)) = global;

                if instrs.len() > 2 {
                    anyhow::bail!("multiple instr global initializers are not supported");
                }

                let global = match instrs.first() {
                    Some(Instr::I32Const(v)) => Value::I32(*v),
                    Some(Instr::I64Const(v)) => Value::I64(*v),
                    Some(Instr::F32Const(v)) => Value::F32(*v),
                    Some(Instr::F64Const(v)) => Value::F64(*v),
                    Some(Instr::GlobalGet(GlobalIdx(idx))) => {
                        let idx = *idx as usize;
                        globals[idx].value()
                    }
                    Some(Instr::RefNull(_ref_type)) => Value::RefNull,
                    Some(Instr::RefFunc(FuncIdx(idx))) => Value::RefFunc(FuncIdx(*idx)),
                    _ => anyhow::bail!("unsupported global initializer instruction"),
                };

                global_type.0.validate(&global).context("global type does not accept this value")?;
                globals.push(GlobalInst {
                    r#type: *global_type,
                    r#impl: GlobalInstImpl::Guest(global),
                });
                Ok(globals)
            })?;

        // next step: build our function table.
        let functions = func_imports
            .iter()
            .map(|desc| {
                if desc.0 as usize
                    >= self
                        .parsed_module
                        .type_section()
                        .map(|xs| xs.len())
                        .unwrap_or_default()
                {
                    anyhow::bail!("type out of range");
                }

                Ok(FuncInst {
                    r#type: *desc,
                    r#impl: FuncInstImpl::Host(TKTK),
                })
            })
            .chain(
                self.parsed_module
                    .function_section()
                    .iter()
                    .flat_map(|xs| xs.iter())
                    .copied()
                    .enumerate()
                    .map(|(code_idx, xs)| {
                        if code_idx
                            >= self
                                .parsed_module
                                .code_section()
                                .map(|xs| xs.len())
                                .unwrap_or_default()
                        {
                            anyhow::bail!("code idx out of range");
                        }

                        Ok(FuncInst {
                            r#type: xs,
                            r#impl: FuncInstImpl::Guest(CodeIdx(code_idx as u32)),
                        })
                    }),
            )
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut memories: Vec<_> = memory_imports
            .iter()
            .map(|desc| MemInst {
                r#type: *desc,
                r#impl: MemoryInstImpl::Host(TKTK),
            })
            .chain(
                self.parsed_module
                    .memory_section()
                    .iter()
                    .flat_map(|xs| xs.iter())
                    .map(|memtype| MemInst {
                        r#type: *memtype,
                        r#impl: MemoryInstImpl::Guest(MemoryRegion::new(memtype.0)),
                    }),
            )
            .collect();

        let mut tables: Vec<_> = table_imports
            .iter()
            .map(|desc| TableInst {
                r#type: *desc,
                r#impl: TableInstImpl::Host(TKTK),
            })
            .chain(
                self.parsed_module
                    .table_section()
                    .iter()
                    .flat_map(|xs| xs.iter())
                    .map(|tabletype| TableInst {
                        r#type: *tabletype,
                        r#impl: TableInstImpl::Guest(vec![
                            Value::RefNull;
                            tabletype.1.min() as usize
                        ]),
                    }),
            )
            .collect();

        // TODO:
        // - [x] apply data to memories
        // - [ ] apply elems to tables
        for data in self
            .parsed_module
            .data_section()
            .iter()
            .flat_map(|xs| xs.iter())
        {
            match data {
                Data::Active(data, memory_idx, expr) => {
                    let offset = compute_constant_expr(expr, globals.as_slice())?;
                    let offset = offset
                        .as_usize()
                        .ok_or_else(|| anyhow::anyhow!("expected i32 or i64"))?;

                    memories[memory_idx.0 as usize].copy_data(data, offset)?;
                }
                Data::Passive(_) => continue,
            }
        }

        for elem in self
            .parsed_module
            .element_section()
            .iter()
            .flat_map(|xs| xs.iter())
        {
            match elem {
                Elem::ActiveSegmentFuncs(expr, func_indices) => {
                    let offset = compute_constant_expr(expr, globals.as_slice())?;
                    let offset = offset
                        .as_usize()
                        .ok_or_else(|| anyhow::anyhow!("expected i32 or i64"))?;

                    let Some(table) = tables.get_mut(offset) else {
                        anyhow::bail!("could not populate elements: no table at idx={}", offset);
                    };

                    match &mut table.r#impl {
                        TableInstImpl::Guest(v) => {
                            for (idx, xs) in func_indices.iter().enumerate() {
                                v[idx] = Value::RefFunc(*xs);
                            }
                        }
                        TableInstImpl::Host(_) => {
                            todo!("TODO: handle populating imported tables with elements")
                        }
                    }
                }
                Elem::PassiveSegment(_, _) => todo!("PassiveSegment"),
                Elem::ActiveSegment(_, _, _, _) => todo!("ActiveSegment"),
                Elem::DeclarativeSegment(_, _) => todo!("DeclarativeSegment"),
                Elem::ActiveSegmentExpr(_, _) => todo!("ActiveSegmentExpr"),
                Elem::PassiveSegmentExpr(_, _) => todo!("PassiveSegmentExpr"),
                Elem::ActiveSegmentTableAndExpr(_, _, _, _) => todo!("ActiveSegmentTableAndExpr"),
                Elem::DeclarativeSegmentExpr(_, _) => todo!("DeclarativeSegmentExpr"),
            }
        }

        let mut map = HashMap::new();
        for exports in self
            .parsed_module
            .export_section()
            .iter()
            .flat_map(|xs| xs.iter())
        {
            map.insert(exports.nm.0, exports.desc);
        }

        Ok(ModuleInstance {
            exports: map,
            module: self,
            functions,
            globals,
            memories,
            tables,
        })
    }
}

impl<'a> ModuleInstance<'a> {
    pub(crate) fn call(&mut self, funcname: &str, args: &[Value]) -> anyhow::Result<Vec<Value>> {
        let ModuleInstance {
            exports,
            module,
            ref functions,
            globals,
            memories,
            tables,
        } = self;

        struct ModuleInstanceLocal<'a> {
            module: &'a Module<'a>,
            exports: &'a HashMap<&'a str, ExportDesc>,
            functions: &'a Vec<FuncInst>,
            globals: &'a mut Vec<GlobalInst>,
            memories: &'a mut Vec<MemInst>,
            tables: &'a mut Vec<TableInst>,
        }

        let instance = ModuleInstanceLocal {
            exports,
            module,
            functions,
            globals,
            memories,
            tables,
        };

        let Some(ExportDesc::Func(func_idx)) = exports.get(funcname) else {
            anyhow::bail!("no such function, {}", funcname);
        };

        let Some(func) = functions.get(func_idx.0 as usize) else {
            anyhow::bail!(
                "no such function, {} (idx {} not in range)",
                funcname,
                func_idx.0
            );
        };

        let FuncInstImpl::Guest(code_idx) = func.r#impl else {
            todo!("imports")
        };

        let Some(typedef) = module
            .parsed_module
            .type_section()
            .map(|types| &types[func.r#type.0 as usize])
        else {
            anyhow::bail!("no type definition for {}", funcname);
        };

        let Some(code) = module
            .parsed_module
            .code_section()
            .iter()
            .flat_map(|xs| xs.iter())
            .nth(code_idx.0 as usize)
        else {
            anyhow::bail!("could not find code for function");
        };

        let param_types = typedef.0.0.as_slice();
        if args.len() < param_types.len() {
            anyhow::bail!(
                "not enough arguments to call {}; expected {} args",
                funcname,
                param_types.len()
            );
        }

        if args.len() > param_types.len() {
            anyhow::bail!(
                "too many arguments to call {}; expected {} args",
                funcname,
                param_types.len()
            );
        }

        for (idx, (param_type, value)) in param_types.iter().zip(args.iter()).enumerate() {
            param_type.validate(value).with_context(|| format!("bad argument at {}", idx))?;
        }

        let FuncInstImpl::Guest(_func_inst_impl) = func.r#impl else {
            todo!("implement re-exported imports");
        };
        let locals = code.0.locals.as_slice();

        let mut locals: Vec<Value> = args
            .iter()
            .cloned()
            .chain(locals.iter().map(|xs| xs.1.instantiate()))
            .collect();

        // TODO: the reference to the ModuleInstance needs to be mostly-immutable but
        // sometimes-mutable.
        struct Frame<'a> {
            #[cfg(test)]
            name: &'static str,
            pc: usize,
            return_unwind_count: usize,
            instrs: &'a [Instr],
            jump_to: Option<usize>,
            block_type: BlockType,
            locals_base_offset: usize,
            instance: Option<ModuleInstanceLocal<'a>>,
        }

        let mut value_stack = Vec::<Value>::new();
        let mut frames = Vec::<Frame<'a>>::new();
        frames.push(Frame {
            #[cfg(test)]
            name: "init",
            pc: 0,
            return_unwind_count: 1,
            instrs: &[],
            jump_to: None,
            locals_base_offset: 0,
            block_type: BlockType::TypeIndex(func.r#type.0 as i32),
            instance: Some(instance),
        });

        frames.push(Frame {
            #[cfg(test)]
            name: "call",
            pc: 0,
            return_unwind_count: 2,
            instrs: code.0.expr.0.as_slice(),
            jump_to: None,
            locals_base_offset: 0,
            block_type: BlockType::TypeIndex(func.r#type.0 as i32),
            instance: None,
        });

        loop {
            let frame_idx = frames.len() - 1;
            if frames[frame_idx].pc >= frames[frame_idx].instrs.len() {
                locals.shrink_to(frames[frame_idx].locals_base_offset);
                let last_frame = frames.pop().expect("we should always be able to pop a frame");

                if frames.is_empty() {
                    break;
                }

                if let Some(instance) = last_frame.instance {
                    let frame_idx = frame_idx - 1;
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    frames[instance_offset].instance.replace(instance);
                }

                continue;
            }

            match &frames[frame_idx].instrs[frames[frame_idx].pc] {
                Instr::Unreachable => anyhow::bail!("hit trap"),
                Instr::Nop => {}
                Instr::Block(block_type, blockinstrs) => {
                    let locals_base_offset = frames[frame_idx].locals_base_offset;
                    frames.push(Frame {
                        #[cfg(test)]
                        name: "block",
                        pc: 0,
                        return_unwind_count: frames[frame_idx].return_unwind_count + 1,
                        instrs: blockinstrs,
                        jump_to: Some(blockinstrs.len()),
                        block_type: *block_type,
                        locals_base_offset,
                        instance: None,
                    });
                }

                Instr::Loop(block_type, blockinstrs) => {
                    frames.push(Frame {
                        #[cfg(test)]
                        name: "Loop",
                        pc: 0,
                        return_unwind_count: frames[frame_idx].return_unwind_count + 1,
                        instrs: blockinstrs,
                        jump_to: Some(0),
                        block_type: *block_type,
                        locals_base_offset: frames[frame_idx].locals_base_offset,
                        instance: None,
                    });
                }

                Instr::If(block_type, consequent) => {
                    if value_stack.is_empty() {
                        anyhow::bail!("expected 1 value on stack");
                    }

                    let blockinstrs = if let Some(Value::I32(0) | Value::I64(0)) = value_stack.pop()
                    {
                        &[]
                    } else {
                        consequent.as_slice()
                    };

                    frames.push(Frame {
                        #[cfg(test)]
                        name: "If",
                        pc: 0,
                        return_unwind_count: frames[frame_idx].return_unwind_count + 1,
                        instrs: blockinstrs,
                        jump_to: Some(blockinstrs.len()),
                        block_type: *block_type,
                        locals_base_offset: frames[frame_idx].locals_base_offset,
                        instance: None,
                    });
                }

                Instr::IfElse(block_type, consequent, alternate) => {
                    if value_stack.is_empty() {
                        anyhow::bail!("expected 1 value on stack");
                    }

                    let blockinstrs = if let Some(Value::I32(0) | Value::I64(0)) = value_stack.pop()
                    {
                        alternate.as_slice()
                    } else {
                        consequent.as_slice()
                    };

                    frames.push(Frame {
                        #[cfg(test)]
                        name: "IfElse",
                        pc: 0,
                        return_unwind_count: frames[frame_idx].return_unwind_count + 1,
                        instrs: blockinstrs,
                        jump_to: Some(blockinstrs.len()),
                        block_type: *block_type,
                        locals_base_offset: frames[frame_idx].locals_base_offset,
                        instance: None,
                    });
                }

                Instr::Br(idx) => {
                    if idx.0 > 0 {
                        frames.truncate(frames.len() - idx.0 as usize);
                    }

                    let frame_idx = frames.len() - 1;
                    let Some(jump_to) = frames[frame_idx].jump_to else {
                        anyhow::bail!("invalid jump target");
                    };
                    frames[frame_idx].pc = jump_to;
                    continue;
                }

                Instr::BrIf(idx) => {
                    let Some(Value::I32(v)) = value_stack.pop() else {
                        anyhow::bail!("expected an i32 value on the stack");
                    };

                    if v != 0 {
                        if idx.0 > 0 {
                            frames.truncate(frames.len() - idx.0 as usize);
                        }

                        let frame_idx = frames.len() - 1;
                        let Some(jump_to) = frames[frame_idx].jump_to else {
                            anyhow::bail!("invalid jump target");
                        };
                        frames[frame_idx].pc = jump_to;
                    }
                    continue;
                }

                Instr::BrTable(labels, alternate) => {
                    let Some(Value::I32(v)) = value_stack.pop() else {
                        anyhow::bail!("expected an i32 value on the stack");
                    };

                    let v = v as usize;
                    let idx = if v >= labels.len() {
                        alternate.0
                    } else {
                        labels[v].0
                    } as usize;

                    if idx > 0 {
                        frames.truncate(frames.len() - idx);
                    }

                    let frame_idx = frames.len() - 1;
                    let Some(jump_to) = frames[frame_idx].jump_to else {
                        anyhow::bail!("invalid jump target");
                    };
                    frames[frame_idx].pc = jump_to;
                    continue;
                }

                Instr::Return => {
                    // TODO: validate result type!
                    frames.truncate(frames.len() - frames[frame_idx].return_unwind_count);
                    if frames.is_empty() {
                        break
                    }

                    let new_frame_idx = frames.len() - 1;
                    frames[new_frame_idx].pc = frames[new_frame_idx].instrs.len();
                    continue
                }

                Instr::Call(func_idx) => {
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    let Some(instance) =
                        frames[instance_offset].instance.take()
                    else {
                        anyhow::bail!("call: could not access instance");
                    };

                    let Some(func) = instance.functions.get(func_idx.0 as usize) else {
                        anyhow::bail!(
                            "no such function, {} (idx {} not in range)",
                            funcname,
                            func_idx.0
                        );
                    };

                    let FuncInstImpl::Guest(code_idx) = func.r#impl else {
                        todo!("imports")
                    };

                    let Some(typedef) = instance
                        .module
                        .parsed_module
                        .type_section()
                        .map(|types| &types[func.r#type.0 as usize])
                    else {
                        anyhow::bail!("no type definition for {}", funcname);
                    };

                    let Some(code) = instance
                        .module
                        .parsed_module
                        .code_section()
                        .iter()
                        .flat_map(|xs| xs.iter())
                        .nth(code_idx.0 as usize)
                    else {
                        anyhow::bail!("could not find code for function");
                    };

                    let param_types = typedef.0 .0.as_slice();
                    if value_stack.len() < param_types.len() {
                        anyhow::bail!(
                            "not enough arguments to call func idx={}; expected {} args",
                            func_idx.0,
                            param_types.len()
                        );
                    }

                    let args = value_stack.split_off(value_stack.len() - param_types.len());

                    let locals_base_offset = locals.len();
                    locals.reserve(code.0.locals.len());

                    for (idx, (param_type, value)) in
                        param_types.iter().zip(args.iter()).enumerate()
                    {
                        param_type.validate(value).with_context(|| format!("bad argument at {}", idx))?;
                        locals.push(*value);
                    }

                    for local in code.0.locals.iter().skip(param_types.len()) {
                        locals.push(local.1.instantiate());
                    }

                    frames.push(Frame {
                        #[cfg(test)]
                        name: "Call",
                        pc: 0,
                        return_unwind_count: 1,
                        instrs: &code.0.expr.0,
                        jump_to: None,
                        locals_base_offset,
                        block_type: BlockType::TypeIndex(func.r#type.0 as i32),
                        instance: Some(ModuleInstanceLocal {
                            module: instance.module,
                            exports: instance.exports,
                            functions: instance.functions,
                            globals: &mut *instance.globals,
                            memories: &mut *instance.memories,
                            tables: &mut *instance.tables,
                        }),
                    });
                }

                Instr::CallIndirect(_type_idx, table_idx) => {
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    let Some(instance) =
                        frames[instance_offset].instance.take()
                    else {
                        anyhow::bail!("call_indirect: could not access instance");
                    };

                    let Some(table) = instance.tables.get(table_idx.0 as usize) else {
                        anyhow::bail!("invalid table id={}", table_idx.0);
                    };

                    let Some(Value::I32(v)) = value_stack.pop() else {
                        anyhow::bail!("expected an i32 value on the stack");
                    };

                    let TableType(_reftype, _limits) = &table.r#type;

                    let v = match &table.r#impl {
                        TableInstImpl::Guest(values) => {
                            let Some(v) = values.get(v as usize) else {
                                anyhow::bail!("undefined element: table index out of range");
                            };
                            v
                        }
                        TableInstImpl::Host(_) => todo!(),
                    };

                    let Value::RefFunc(v) = v else {
                        anyhow::bail!("expected reffunc value, got {:?}", v);
                    };

                    let Some(func) = instance.functions.get(v.0 as usize) else {
                        anyhow::bail!(
                            "no such function, {} (idx {} not in range)",
                            funcname,
                            func_idx.0
                        );
                    };

                    let FuncInstImpl::Guest(code_idx) = func.r#impl else {
                        todo!("imports")
                    };

                    let Some(typedef) = instance
                        .module
                        .parsed_module
                        .type_section()
                        .map(|types| &types[func.r#type.0 as usize])
                    else {
                        anyhow::bail!("no type definition for {}", funcname);
                    };

                    let Some(code) = instance
                        .module
                        .parsed_module
                        .code_section()
                        .iter()
                        .flat_map(|xs| xs.iter())
                        .nth(code_idx.0 as usize)
                    else {
                        anyhow::bail!("could not find code for function");
                    };

                    let param_types = typedef.0 .0.as_slice();
                    if value_stack.len() < param_types.len() {
                        anyhow::bail!(
                            "not enough arguments to call func idx={}; expected {} args",
                            func_idx.0,
                            param_types.len()
                        );
                    }

                    let args = value_stack.split_off(value_stack.len() - param_types.len());

                    let locals_base_offset = locals.len();
                    locals.reserve(code.0.locals.len());

                    for (idx, (param_type, value)) in
                        param_types.iter().zip(args.iter()).enumerate()
                    {
                        param_type.validate(value).with_context(|| format!("bad argument at {}", idx))?;
                        locals.push(*value);
                    }

                    for local in code.0.locals.iter().skip(param_types.len()) {
                        locals.push(local.1.instantiate());
                    }

                    frames.push(Frame {
                        #[cfg(test)]
                        name: "CallIndirect",
                        pc: 0,
                        return_unwind_count: 1,
                        instrs: &code.0.expr.0,
                        jump_to: None,
                        locals_base_offset,
                        block_type: BlockType::TypeIndex(func.r#type.0 as i32),
                        instance: Some(instance),
                    });
                }

                Instr::RefNull(_) => todo!("RefNull"),
                Instr::RefIsNull => todo!("RefIsNull"),
                Instr::RefFunc(_) => todo!("RefFunc"),
                Instr::Drop => {
                    let Some(_) = value_stack.pop() else {
                        anyhow::bail!("drop out of range")
                    };
                }
                Instr::SelectEmpty => {
                    let items = value_stack.split_off(value_stack.len() - 3);
                    value_stack.push(if items[2].is_zero()? {
                        items[1]
                    } else {
                        items[0]
                    });
                }

                Instr::Select(_) => todo!("Select"),

                Instr::LocalGet(idx) => {
                    let Some(v) = locals.get(idx.0 as usize + frames[frame_idx].locals_base_offset)
                    else {
                        anyhow::bail!("local.get out of range")
                    };

                    value_stack.push(*v);
                }

                Instr::LocalSet(idx) => {
                    locals[idx.0 as usize + frames[frame_idx].locals_base_offset] = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("ran out of stack"))?;
                }

                Instr::LocalTee(idx) => {
                    locals[idx.0 as usize + frames[frame_idx].locals_base_offset] = *value_stack
                        .last()
                        .ok_or_else(|| anyhow::anyhow!("ran out of stack"))?;
                }

                Instr::GlobalGet(global_idx) => {
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    let Some(instance) =
                        frames[instance_offset].instance.as_ref()
                    else {
                        anyhow::bail!("could not access instance");
                    };

                    // TODO: respect base globals offset
                    let Some(global) = instance.globals.get(global_idx.0 as usize) else {
                        anyhow::bail!("global idx out of range");
                    };

                    match global.r#impl {
                        GlobalInstImpl::Guest(lhs) => value_stack.push(lhs),
                        GlobalInstImpl::Host(_) => todo!(),
                    }
                }


                Instr::GlobalSet(global_idx) => {
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    let Some(instance) =
                        frames[instance_offset].instance.take()
                    else {
                        anyhow::bail!("call: could not access instance");
                    };

                    // TODO: respect base globals offset
                    let Some(global) = instance.globals.get_mut(global_idx.0 as usize) else {
                        anyhow::bail!("global idx out of range");
                    };

                    if global.r#type.1 == Mutability::Const {
                        anyhow::bail!("call: could not access instance");
                    }

                    let Some(v) = value_stack.pop() else {
                        anyhow::bail!("drop out of range")
                    };

                    global.r#type.0.validate(&v).context("global type value type mismatch")?;

                    match &mut global.r#impl {
                        GlobalInstImpl::Guest(lhs) => *lhs = v,
                        GlobalInstImpl::Host(_) => todo!(),
                    }
                    frames[instance_offset].instance.replace(instance);
                },

                Instr::TableGet(_) => todo!("TableGet"),
                Instr::TableSet(_) => todo!("TableSet"),
                Instr::TableInit(_, _) => todo!("TableInit"),
                Instr::ElemDrop(_) => todo!("ElemDrop"),
                Instr::TableCopy(_, _) => todo!("TableCopy"),
                Instr::TableGrow(_) => todo!("TableGrow"),
                Instr::TableSize(_) => todo!("TableSize"),
                Instr::TableFill(_) => todo!("TableFill"),

                Instr::I32Load(mem) => {
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    let Some(instance) =
                        frames[instance_offset].instance.as_ref()
                    else {
                        anyhow::bail!("could not access instance");
                    };

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = instance.memories[mem.memidx()].load(offset)?;
                    value_stack.push(Value::I32(i32::from_le_bytes(arr)));
                }

                Instr::I64Load(mem) => {
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    let Some(instance) =
                        frames[instance_offset].instance.as_ref()
                    else {
                        anyhow::bail!("could not access instance");
                    };

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = instance.memories[mem.memidx()].load(offset)?;
                    value_stack.push(Value::I64(i64::from_le_bytes(arr)));
                }

                Instr::F32Load(mem) => {
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    let Some(instance) =
                        frames[instance_offset].instance.as_ref()
                    else {
                        anyhow::bail!("could not access instance");
                    };

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = instance.memories[mem.memidx()].load(offset)?;
                    value_stack.push(Value::F32(f32::from_le_bytes(arr)));
                }

                Instr::F64Load(mem) => {
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    let Some(instance) =
                        frames[instance_offset].instance.as_ref()
                    else {
                        anyhow::bail!("could not access instance");
                    };

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = instance.memories[mem.memidx()].load(offset)?;
                    value_stack.push(Value::F64(f64::from_le_bytes(arr)));
                }

                Instr::I32Load8S(mem) => {
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    let Some(instance) =
                        frames[instance_offset].instance.as_ref()
                    else {
                        anyhow::bail!("could not access instance");
                    };

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = instance.memories[mem.memidx()].load::<1>(offset)?;
                    value_stack.push(Value::I32(arr[0] as i32));
                }

                Instr::I32Load8U(mem) => {
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    let Some(instance) =
                        frames[instance_offset].instance.as_ref()
                    else {
                        anyhow::bail!("could not access instance");
                    };

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = instance.memories[mem.memidx()].load::<1>(offset)?;
                    value_stack.push(Value::I32(arr[0] as u32 as i32));
                }

                Instr::I32Load16S(mem) => {
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    let Some(instance) =
                        frames[instance_offset].instance.as_ref()
                    else {
                        anyhow::bail!("could not access instance");
                    };

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = instance.memories[mem.memidx()].load(offset)?;
                    value_stack.push(Value::I32(i16::from_le_bytes(arr) as i32));
                }

                Instr::I32Load16U(mem) => {
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    let Some(instance) =
                        frames[instance_offset].instance.as_ref()
                    else {
                        anyhow::bail!("could not access instance");
                    };

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = instance.memories[mem.memidx()].load(offset)?;
                    value_stack.push(Value::I32(u16::from_le_bytes(arr) as i32));
                }

                Instr::I64Load8S(mem) => {
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    let Some(instance) =
                        frames[instance_offset].instance.as_ref()
                    else {
                        anyhow::bail!("could not access instance");
                    };

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = instance.memories[mem.memidx()].load::<1>(offset)?;
                    value_stack.push(Value::I64(arr[0] as i64));
                }

                Instr::I64Load8U(mem) => {
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    let Some(instance) =
                        frames[instance_offset].instance.as_ref()
                    else {
                        anyhow::bail!("could not access instance");
                    };

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = instance.memories[mem.memidx()].load::<1>(offset)?;
                    value_stack.push(Value::I64(arr[0] as u64 as i64));
                }

                Instr::I64Load16S(mem) => {
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    let Some(instance) =
                        frames[instance_offset].instance.as_ref()
                    else {
                        anyhow::bail!("could not access instance");
                    };

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = instance.memories[mem.memidx()].load(offset)?;
                    value_stack.push(Value::I64(i16::from_le_bytes(arr) as i64));
                }

                Instr::I64Load16U(mem) => {
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    let Some(instance) =
                        frames[instance_offset].instance.as_ref()
                    else {
                        anyhow::bail!("could not access instance");
                    };

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = instance.memories[mem.memidx()].load(offset)?;
                    value_stack.push(Value::I64(u16::from_le_bytes(arr) as i64));
                }

                Instr::I64Load32S(mem) => {
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    let Some(instance) =
                        frames[instance_offset].instance.as_ref()
                    else {
                        anyhow::bail!("could not access instance");
                    };

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = instance.memories[mem.memidx()].load(offset)?;
                    value_stack.push(Value::I64(i32::from_le_bytes(arr) as i64));
                }

                Instr::I64Load32U(mem) => {
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    let Some(instance) =
                        frames[instance_offset].instance.as_ref()
                    else {
                        anyhow::bail!("could not access instance");
                    };

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = instance.memories[mem.memidx()].load(offset)?;
                    value_stack.push(Value::I64(u32::from_le_bytes(arr) as i64));
                }

                Instr::I32Store(mem) => {
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    let Some(instance) =
                        frames[instance_offset].instance.take()
                    else {
                        anyhow::bail!("could not access instance");
                    };

                    let Some(Value::I32(v)) = value_stack.pop() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };

                    let Some(Value::I32(t)) = value_stack.pop() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };

                    let offset = mem.offset().saturating_add(t as usize);
                    instance.memories[mem.memidx()].write(offset, &v.to_le_bytes())?;
                    frames[instance_offset].instance.replace(instance);
                },

                Instr::I64Store(_) => todo!("I64Store"),
                Instr::F32Store(_) => todo!("F32Store"),
                Instr::F64Store(_) => todo!("F64Store"),
                Instr::I32Store8(_) => todo!("I32Store8"),
                Instr::I32Store16(_) => todo!("I32Store16"),
                Instr::I64Store8(_) => todo!("I64Store8"),
                Instr::I64Store16(_) => todo!("I64Store16"),
                Instr::I64Store32(_) => todo!("I64Store32"),
                Instr::MemorySize(_) => todo!("MemorySize"),
                Instr::MemoryGrow(mem_idx) => {
                    let instance_offset = frames.len() - frames[frame_idx].return_unwind_count;
                    let Some(instance) =
                        frames[instance_offset].instance.take()
                    else {
                        anyhow::bail!("could not access instance");
                    };

                    let Some(Value::I32(v)) = value_stack.pop() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };

                    let memory = &mut instance.memories[mem_idx.0 as usize];
                    let page_count = memory.grow(v as usize)?;
                    value_stack.push(Value::I32(page_count as i32));
                    frames[instance_offset].instance.replace(instance);
                },
                Instr::MemoryInit(_, _) => todo!("MemoryInit"),
                Instr::DataDrop(_) => todo!("DataDrop"),
                Instr::MemoryCopy(_, _) => todo!("MemoryCopy"),
                Instr::MemoryFill(_) => todo!("MemoryFill"),
                Instr::I32Const(v) => {
                    value_stack.push(Value::I32(*v));
                }
                Instr::I64Const(v) => {
                    value_stack.push(Value::I64(*v));
                }
                Instr::F32Const(v) => {
                    value_stack.push(Value::F32(*v));
                }
                Instr::F64Const(v) => {
                    value_stack.push(Value::F64(*v));
                }
                Instr::I32Eqz => {
                    let Some(Value::I32(v)) = value_stack.pop() else {
                        anyhow::bail!("i32.eqz: expected 1 i32 value on stack");
                    };
                    value_stack.push(Value::I32(if v == 0 { 1 } else { 0 }));
                }
                Instr::I32Eq => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.Eq: not enough operands");
                    };
                    value_stack.push(Value::I32(if lhs == rhs { 1 } else { 0 }));
                }
                Instr::I32Ne => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.Ne: not enough operands");
                    };
                    value_stack.push(Value::I32(if lhs != rhs { 1 } else { 0 }));
                }
                Instr::I32LtS => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.LtS: not enough operands");
                    };
                    value_stack.push(Value::I32(if lhs < rhs { 1 } else { 0 }));
                }
                Instr::I32LtU => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.LtU: not enough operands");
                    };
                    value_stack.push(Value::I32(if (lhs as u32) < (rhs as u32) { 1 } else { 0 }));
                }

                Instr::I32GtS => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.GtS: not enough operands");
                    };
                    value_stack.push(Value::I32(if lhs > rhs { 1 } else { 0 }));
                }

                Instr::I32GtU => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.GtU: not enough operands");
                    };
                    value_stack.push(Value::I32(if (lhs as u32) > (rhs as u32) { 1 } else { 0 }));
                }
                Instr::I32LeS => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.LeS: not enough operands");
                    };
                    value_stack.push(Value::I32(if lhs <= rhs { 1 } else { 0 }));
                }
                Instr::I32LeU => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.LeU: not enough operands");
                    };
                    value_stack.push(Value::I32(if (lhs as u32) <= (rhs as u32) { 1 } else { 0 }));
                }
                Instr::I32GeS => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.GeS: not enough operands");
                    };
                    value_stack.push(Value::I32(if lhs >= rhs { 1 } else { 0 }));
                }
                Instr::I32GeU => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.GeU: not enough operands");
                    };
                    value_stack.push(Value::I32(if (lhs as u32) >= (rhs as u32) { 1 } else { 0 }));
                }
                Instr::I64Eqz => {
                    let Some(Value::I64(v)) = value_stack.pop() else {
                        anyhow::bail!("i64.eqz: expected 1 i64 value on stack");
                    };
                    value_stack.push(Value::I64(if v == 0 { 1 } else { 0 }));
                }
                Instr::I64Eq => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.Eq: not enough operands");
                    };
                    value_stack.push(Value::I64(if lhs == rhs { 1 } else { 0 }));
                }
                Instr::I64Ne => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.Ne: not enough operands");
                    };
                    value_stack.push(Value::I64(if lhs != rhs { 1 } else { 0 }));
                }
                Instr::I64LtS => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.LtS: not enough operands");
                    };
                    value_stack.push(Value::I64(if lhs < rhs { 1 } else { 0 }));
                }
                Instr::I64LtU => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.LtU: not enough operands");
                    };
                    value_stack.push(Value::I64(if (lhs as u64) < (rhs as u64) { 1 } else { 0 }));
                }

                Instr::I64GtS => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.GtS: not enough operands");
                    };
                    value_stack.push(Value::I64(if lhs > rhs { 1 } else { 0 }));
                }

                Instr::I64GtU => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.GtU: not enough operands");
                    };
                    value_stack.push(Value::I64(if (lhs as u64) > (rhs as u64) { 1 } else { 0 }));
                }
                Instr::I64LeS => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.LeS: not enough operands");
                    };
                    value_stack.push(Value::I64(if lhs <= rhs { 1 } else { 0 }));
                }
                Instr::I64LeU => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.LeU: not enough operands");
                    };
                    value_stack.push(Value::I64(if (lhs as u64) <= (rhs as u64) { 1 } else { 0 }));
                }
                Instr::I64GeS => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.GeS: not enough operands");
                    };
                    value_stack.push(Value::I64(if lhs >= rhs { 1 } else { 0 }));
                }
                Instr::I64GeU => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.GeU: not enough operands");
                    };
                    value_stack.push(Value::I64(if (lhs as u64) >= (rhs as u64) { 1 } else { 0 }));
                }
                Instr::F32Eq => {
                    let (Some(Value::F32(rhs)), Some(Value::F32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f32.eq: not enough operands");
                    };
                    value_stack.push(Value::I32(if lhs == rhs { 1 } else { 0 }));
                }
                Instr::F32Ne => {
                    let (Some(Value::F32(rhs)), Some(Value::F32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f32.ne: not enough operands");
                    };
                    value_stack.push(Value::I32(if lhs != rhs { 1 } else { 0 }));
                }
                Instr::F32Lt => {
                    let (Some(Value::F32(rhs)), Some(Value::F32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f32.lt: not enough operands");
                    };
                    value_stack.push(Value::I32(if lhs < rhs { 1 } else { 0 }));
                }
                Instr::F32Gt => {
                    let (Some(Value::F32(rhs)), Some(Value::F32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f32.gt: not enough operands");
                    };
                    value_stack.push(Value::I32(if lhs > rhs { 1 } else { 0 }));
                }
                Instr::F32Le => {
                    let (Some(Value::F32(rhs)), Some(Value::F32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f32.le: not enough operands");
                    };
                    value_stack.push(Value::I32(if lhs <= rhs { 1 } else { 0 }));
                }
                Instr::F32Ge => {
                    let (Some(Value::F32(rhs)), Some(Value::F32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f32.ge: not enough operands");
                    };
                    value_stack.push(Value::I32(if lhs >= rhs { 1 } else { 0 }));
                }
                Instr::F64Eq => {
                    let (Some(Value::F64(rhs)), Some(Value::F64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f64.eq: not enough operands");
                    };
                    value_stack.push(Value::I32(if lhs == rhs { 1 } else { 0 }));
                }
                Instr::F64Ne => {
                    let (Some(Value::F64(rhs)), Some(Value::F64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f64.ne: not enough operands");
                    };
                    value_stack.push(Value::I32(if lhs != rhs { 1 } else { 0 }));
                }
                Instr::F64Lt => {
                    let (Some(Value::F64(rhs)), Some(Value::F64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f64.lt: not enough operands");
                    };
                    value_stack.push(Value::I32(if lhs < rhs { 1 } else { 0 }));
                }
                Instr::F64Gt => {
                    let (Some(Value::F64(rhs)), Some(Value::F64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f64.gt: not enough operands");
                    };
                    value_stack.push(Value::I32(if lhs > rhs { 1 } else { 0 }));
                }
                Instr::F64Le => {
                    let (Some(Value::F64(rhs)), Some(Value::F64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f64.le: not enough operands");
                    };
                    value_stack.push(Value::I32(if lhs <= rhs { 1 } else { 0 }));
                }
                Instr::F64Ge => {
                    let (Some(Value::F64(rhs)), Some(Value::F64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f64.ge: not enough operands");
                    };
                    value_stack.push(Value::I32(if lhs >= rhs { 1 } else { 0 }));
                }
                Instr::I32Clz => {
                    let Some(Value::I32(op)) = value_stack.pop() else {
                        anyhow::bail!("i32.clz: not enough operands");
                    };
                    value_stack.push(Value::I32(op.leading_zeros() as i32));
                }
                Instr::I32Ctz => {
                    let Some(Value::I32(op)) = value_stack.pop() else {
                        anyhow::bail!("i32.ctz: not enough operands");
                    };
                    value_stack.push(Value::I32(op.trailing_zeros() as i32));
                }
                Instr::I32Popcnt => {
                    let Some(Value::I32(op)) = value_stack.pop() else {
                        anyhow::bail!("i32.ctz: not enough operands");
                    };
                    value_stack.push(Value::I32(op.count_ones() as i32));
                }
                Instr::I32Add => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.add: not enough operands");
                    };
                    value_stack.push(Value::I32(lhs + rhs));
                }
                Instr::I32Sub => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.sub: not enough operands");
                    };
                    value_stack.push(Value::I32(lhs - rhs));
                }
                Instr::I32Mul => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.mul: not enough operands");
                    };
                    value_stack.push(Value::I32(lhs * rhs));
                }
                Instr::I32DivS => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.divS: not enough operands");
                    };
                    value_stack.push(Value::I32(lhs / rhs));
                }
                Instr::I32DivU => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.divU: not enough operands");
                    };
                    value_stack.push(Value::I32(((lhs as u32) / (rhs as u32)) as i32));
                }
                Instr::I32RemS => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.remS: not enough operands");
                    };
                    value_stack.push(Value::I32(lhs % rhs));
                }
                Instr::I32RemU => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.remU: not enough operands");
                    };
                    value_stack.push(Value::I32(((lhs as u32) % (rhs as u32)) as i32));
                }
                Instr::I32And => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.and: not enough operands");
                    };
                    value_stack.push(Value::I32(lhs & rhs));
                }
                Instr::I32Ior => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.ior: not enough operands");
                    };
                    value_stack.push(Value::I32(lhs | rhs));
                }
                Instr::I32Xor => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.xor: not enough operands");
                    };
                    value_stack.push(Value::I32(lhs ^ rhs));
                }
                Instr::I32Shl => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.shl: not enough operands");
                    };
                    value_stack.push(Value::I32(lhs << rhs));
                }
                Instr::I32ShrS => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.shrS: not enough operands");
                    };
                    value_stack.push(Value::I32(lhs >> rhs));
                }
                Instr::I32ShrU => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.shrU: not enough operands");
                    };
                    value_stack.push(Value::I32(((lhs as u32) >> rhs) as i32));
                }
                Instr::I32Rol => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.rol: not enough operands");
                    };
                    value_stack.push(Value::I32(lhs.rotate_left(rhs as u32)));
                }
                Instr::I32Ror => {
                    let (Some(Value::I32(rhs)), Some(Value::I32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i32.ror: not enough operands");
                    };
                    value_stack.push(Value::I32(lhs.rotate_right(rhs as u32)));
                }

                Instr::I64Clz => {
                    let Some(Value::I64(op)) = value_stack.pop() else {
                        anyhow::bail!("i64.clz: not enough operands");
                    };
                    value_stack.push(Value::I64(op.leading_zeros() as i64));
                }

                Instr::I64Ctz => {
                    let Some(Value::I64(op)) = value_stack.pop() else {
                        anyhow::bail!("i64.ctz: not enough operands");
                    };
                    value_stack.push(Value::I64(op.trailing_zeros() as i64));
                }

                Instr::I64Popcnt => {
                    let Some(Value::I64(op)) = value_stack.pop() else {
                        anyhow::bail!("i64.ctz: not enough operands");
                    };
                    value_stack.push(Value::I64(op.count_ones() as i64));
                }

                Instr::I64Add => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.add: not enough operands");
                    };
                    value_stack.push(Value::I64(lhs + rhs));
                }

                Instr::I64Sub => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.sub: not enough operands {:?}", value_stack);
                    };
                    value_stack.push(Value::I64(lhs - rhs));
                }

                Instr::I64Mul => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.mul: not enough operands {:?}", value_stack);
                    };
                    value_stack.push(Value::I64(lhs.wrapping_mul(rhs)));
                }
                Instr::I64DivS => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.divS: not enough operands");
                    };
                    value_stack.push(Value::I64(lhs / rhs));
                }
                Instr::I64DivU => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.divU: not enough operands");
                    };
                    value_stack.push(Value::I64(((lhs as u64) / (rhs as u64)) as i64));
                }
                Instr::I64RemS => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.remS: not enough operands");
                    };
                    value_stack.push(Value::I64(lhs % rhs));
                }
                Instr::I64RemU => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.remU: not enough operands");
                    };
                    value_stack.push(Value::I64(((lhs as u64) % (rhs as u64)) as i64));
                }
                Instr::I64And => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.and: not enough operands");
                    };
                    value_stack.push(Value::I64(lhs & rhs));
                }
                Instr::I64Ior => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.ior: not enough operands");
                    };
                    value_stack.push(Value::I64(lhs | rhs));
                }
                Instr::I64Xor => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.xor: not enough operands");
                    };
                    value_stack.push(Value::I64(lhs ^ rhs));
                }
                Instr::I64Shl => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.shl: not enough operands");
                    };
                    value_stack.push(Value::I64(lhs << rhs));
                }
                Instr::I64ShrS => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.shrS: not enough operands");
                    };
                    value_stack.push(Value::I64(lhs >> rhs));
                }
                Instr::I64ShrU => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.shrU: not enough operands");
                    };
                    value_stack.push(Value::I64(((lhs as u64) >> rhs) as i64));
                }
                Instr::I64Rol => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.rol: not enough operands");
                    };
                    value_stack.push(Value::I64(lhs.rotate_left(rhs as u32)));
                }
                Instr::I64Ror => {
                    let (Some(Value::I64(rhs)), Some(Value::I64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("i64.ror: not enough operands");
                    };
                    value_stack.push(Value::I64(lhs.rotate_right(rhs as u32)));
                }

                Instr::F32Abs => {
                    let Some(Value::F32(op)) = value_stack.pop() else {
                        anyhow::bail!("f32.abs: not enough operands");
                    };
                    value_stack.push(Value::F32(op.abs()));
                }
                Instr::F32Neg => {
                    let Some(Value::F32(op)) = value_stack.pop() else {
                        anyhow::bail!("f32.neg: not enough operands");
                    };
                    value_stack.push(Value::F32(-op));
                }
                Instr::F32Ceil => {
                    let Some(Value::F32(op)) = value_stack.pop() else {
                        anyhow::bail!("f32.ceil: not enough operands");
                    };
                    value_stack.push(Value::F32(op.ceil()));
                }
                Instr::F32Floor => {
                    let Some(Value::F32(op)) = value_stack.pop() else {
                        anyhow::bail!("f32.floor: not enough operands");
                    };
                    value_stack.push(Value::F32(op.floor()));
                }
                Instr::F32Trunc => {
                    let Some(Value::F32(op)) = value_stack.pop() else {
                        anyhow::bail!("f32.trunc: not enough operands");
                    };
                    value_stack.push(Value::F32(op.trunc()));
                }
                Instr::F32NearestInt => {
                    let Some(Value::F32(op)) = value_stack.pop() else {
                        anyhow::bail!("f32.nearestInt: not enough operands");
                    };
                    value_stack.push(Value::F32(op.round()));
                }
                Instr::F32Sqrt => {
                    let Some(Value::F32(op)) = value_stack.pop() else {
                        anyhow::bail!("f32.sqrt: not enough operands");
                    };
                    value_stack.push(Value::F32(op.sqrt()));
                }
                Instr::F32Add => {
                    let (Some(Value::F32(rhs)), Some(Value::F32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f32.add: not enough operands");
                    };
                    value_stack.push(Value::F32(lhs + rhs));
                }
                Instr::F32Sub => {
                    let (Some(Value::F32(rhs)), Some(Value::F32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f32.sub: not enough operands");
                    };
                    value_stack.push(Value::F32(lhs - rhs));
                }
                Instr::F32Mul => {
                    let (Some(Value::F32(rhs)), Some(Value::F32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f32.mul: not enough operands");
                    };
                    value_stack.push(Value::F32(lhs * rhs));
                }
                Instr::F32Div => {
                    let (Some(Value::F32(rhs)), Some(Value::F32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f32.div: not enough operands");
                    };
                    value_stack.push(Value::F32(lhs / rhs));
                }
                Instr::F32Min => {
                    let (Some(Value::F32(rhs)), Some(Value::F32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f32.min: not enough operands");
                    };
                    value_stack.push(Value::F32(lhs.min(rhs)));
                }
                Instr::F32Max => {
                    let (Some(Value::F32(rhs)), Some(Value::F32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f32.max: not enough operands");
                    };
                    value_stack.push(Value::F32(lhs.max(rhs)));
                }
                Instr::F32CopySign => {
                    let (Some(Value::F32(rhs)), Some(Value::F32(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f32.copySign: not enough operands");
                    };

                    value_stack.push(Value::F32(
                        match (lhs.is_sign_positive(), rhs.is_sign_positive()) {
                            (true, true) | (true, false) => rhs,
                            (false, true) | (false, false) => -rhs,
                        },
                    ));
                }
                Instr::F64Abs => {
                    let Some(Value::F64(op)) = value_stack.pop() else {
                        anyhow::bail!("f64.abs: not enough operands");
                    };
                    value_stack.push(Value::F64(op.abs()));
                }
                Instr::F64Neg => {
                    let Some(Value::F64(op)) = value_stack.pop() else {
                        anyhow::bail!("f64.neg: not enough operands");
                    };
                    value_stack.push(Value::F64(-op));
                }
                Instr::F64Ceil => {
                    let Some(Value::F64(op)) = value_stack.pop() else {
                        anyhow::bail!("f64.ceil: not enough operands");
                    };
                    value_stack.push(Value::F64(op.ceil()));
                }
                Instr::F64Floor => {
                    let Some(Value::F64(op)) = value_stack.pop() else {
                        anyhow::bail!("f64.floor: not enough operands");
                    };
                    value_stack.push(Value::F64(op.floor()));
                }
                Instr::F64Trunc => {
                    let Some(Value::F64(op)) = value_stack.pop() else {
                        anyhow::bail!("f64.trunc: not enough operands");
                    };
                    value_stack.push(Value::F64(op.trunc()));
                }
                Instr::F64NearestInt => {
                    let Some(Value::F64(op)) = value_stack.pop() else {
                        anyhow::bail!("f64.nearestInt: not enough operands");
                    };
                    value_stack.push(Value::F64(op.round()));
                }
                Instr::F64Sqrt => {
                    let Some(Value::F64(op)) = value_stack.pop() else {
                        anyhow::bail!("f64.sqrt: not enough operands");
                    };
                    value_stack.push(Value::F64(op.sqrt()));
                }
                Instr::F64Add => {
                    let (Some(Value::F64(rhs)), Some(Value::F64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f64.add: not enough operands");
                    };
                    value_stack.push(Value::F64(lhs + rhs));
                }
                Instr::F64Sub => {
                    let (Some(Value::F64(rhs)), Some(Value::F64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f64.sub: not enough operands");
                    };
                    value_stack.push(Value::F64(lhs - rhs));
                }
                Instr::F64Mul => {
                    let (Some(Value::F64(rhs)), Some(Value::F64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f64.mul: not enough operands");
                    };
                    value_stack.push(Value::F64(lhs * rhs));
                }
                Instr::F64Div => {
                    let (Some(Value::F64(rhs)), Some(Value::F64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f64.div: not enough operands");
                    };
                    value_stack.push(Value::F64(lhs / rhs));
                }
                Instr::F64Min => {
                    let (Some(Value::F64(rhs)), Some(Value::F64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f64.min: not enough operands");
                    };
                    value_stack.push(Value::F64(lhs.min(rhs)));
                }
                Instr::F64Max => {
                    let (Some(Value::F64(rhs)), Some(Value::F64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f64.max: not enough operands");
                    };
                    value_stack.push(Value::F64(lhs.max(rhs)));
                }
                Instr::F64CopySign => {
                    let (Some(Value::F64(rhs)), Some(Value::F64(lhs))) =
                        (value_stack.pop(), value_stack.pop())
                    else {
                        anyhow::bail!("f64.copySign: not enough operands");
                    };

                    value_stack.push(Value::F64(
                        match (lhs.is_sign_positive(), rhs.is_sign_positive()) {
                            (true, true) | (true, false) => rhs,
                            (false, true) | (false, false) => -rhs,
                        },
                    ));
                }

                Instr::I32ConvertI64 => todo!("I32ConvertI64"),
                Instr::I32SConvertF32 => todo!("I32SConvertF32"),
                Instr::I32UConvertF32 => todo!("I32UConvertF32"),
                Instr::I32SConvertF64 => todo!("I32SConvertF64"),
                Instr::I32UConvertF64 => todo!("I32UConvertF64"),
                Instr::I64SConvertI32 => {
                    let Some(Value::I32(op)) = value_stack.pop() else {
                        anyhow::bail!("i64.extend_i32_s: not enough operands");
                    };

                    value_stack.push(Value::I64(op as i64));
                },
                Instr::I64UConvertI32 => todo!("I64UConvertI32"),
                Instr::I64SConvertF32 => todo!("I64SConvertF32"),
                Instr::I64UConvertF32 => todo!("I64UConvertF32"),
                Instr::I64SConvertF64 => todo!("I64SConvertF64"),
                Instr::I64UConvertF64 => todo!("I64UConvertF64"),
                Instr::F32SConvertI32 => todo!("F32SConvertI32"),
                Instr::F32UConvertI32 => todo!("F32UConvertI32"),
                Instr::F32SConvertI64 => todo!("F32SConvertI64"),
                Instr::F32UConvertI64 => todo!("F32UConvertI64"),
                Instr::F32ConvertF64 => todo!("F32ConvertF64"),
                Instr::F64SConvertI32 => todo!("F64SConvertI32"),
                Instr::F64UConvertI32 => todo!("F64UConvertI32"),
                Instr::F64SConvertI64 => todo!("F64SConvertI64"),
                Instr::F64UConvertI64 => todo!("F64UConvertI64"),
                Instr::F64ConvertF32 => todo!("F64ConvertF32"),
                Instr::I32ReinterpretF32 => todo!("I32ReinterpretF32"),
                Instr::I64ReinterpretF64 => todo!("I64ReinterpretF64"),
                Instr::F32ReinterpretI32 => todo!("F32ReinterpretI32"),
                Instr::F64ReinterpretI64 => todo!("F64ReinterpretI64"),
                Instr::I32SExtendI8 => todo!("I32SExtendI8"),
                Instr::I32SExtendI16 => todo!("I32SExtendI16"),
                Instr::I64SExtendI8 => todo!("I64SExtendI8"),
                Instr::I64SExtendI16 => todo!("I64SExtendI16"),
                Instr::I64SExtendI32 => todo!("I64SExtendI32"),
            }
            frames[frame_idx].pc += 1;
        }

        // TODO: handle multiple return values
        Ok(value_stack)
    }
}

fn compute_constant_expr(expr: &Expr, globals: &[GlobalInst]) -> anyhow::Result<Value> {
    Ok(match expr.0.first() {
        Some(Instr::F32Const(c)) => Value::F32(*c),
        Some(Instr::F64Const(c)) => Value::F64(*c),
        Some(Instr::I32Const(c)) => Value::I32(*c),
        Some(Instr::I64Const(c)) => Value::I64(*c),
        Some(Instr::GlobalGet(c)) => {
            let global = &globals[c.0 as usize];
            match &global.r#impl {
                GlobalInstImpl::Guest(value) => *value,
                GlobalInstImpl::Host(_tktk) => todo!(),
            }
        }

        Some(Instr::RefNull(_c)) => todo!(),
        Some(Instr::RefFunc(c)) => Value::RefFunc(*c),
        _ => anyhow::bail!("unsupported instruction"),
    })
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::parse::parse;

    #[test]
    fn test_create_store() {
        let bytes = include_bytes!("../example2.wasm");

        let wasm = parse(bytes).unwrap();

        let module = Module::new(wasm);
        let imports = Imports::new(vec![]);
        let mut instance = module
            .instantiate(&imports)
            .expect("could not instantiate module");
        let result = instance
            .call("add_i32", &[Value::I32(1), Value::I32(4)])
            .expect("math is hard");
        assert_eq!(result.as_slice(), &[Value::I32(5)]);
        // dbg!(xs);
    }
}
