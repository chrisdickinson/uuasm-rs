#![allow(dead_code)]
#![allow(clippy::upper_case_acronyms)]

pub(crate) mod function;
pub(crate) mod global;
pub(crate) mod imports;
pub(crate) mod machine;
pub(crate) mod memory;
pub(crate) mod table;
pub(crate) mod value;

pub(crate) use imports::Imports;
pub(crate) use value::Value;

pub(crate) use machine::Machine;
use crate::nodes::{BlockType, NumType, RefType, ValType, VecType};

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
            | (ValType::RefType(RefType::ExternRef), Value::RefNull) => Ok(()),
            (vt, v) => anyhow::bail!("expected={:?}; got={:?}", vt, v),
        }
    }
}

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

#[derive(Debug, Clone, Copy)]
pub(crate) struct TKTK;

#[cfg(test)]
mod test {
    
    use crate::{parse::parse};

    #[test]
    fn test_create_store() {
        let bytes = include_bytes!("../example2.wasm");

        let _wasm = parse(bytes).unwrap();

        /*
        let mut instance = module
            .instantiate(&imports)
            .expect("could not instantiate module");
        let result = instance
            .call("add_i32", &[Value::I32(1), Value::I32(4)])
            .expect("math is hard");
        assert_eq!(result.as_slice(), &[Value::I32(5)]);
        // dbg!(xs);
        */
    }
}
