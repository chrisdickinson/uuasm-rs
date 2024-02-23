#![allow(dead_code)]
#![allow(clippy::upper_case_acronyms)]

use std::collections::HashMap;

use crate::{nodes::{
    Expr, FuncIdx, Global, Instr, LocalIdx, MemType,
    Module as ParsedModule, NumType, RefType, TableType, TypeIdx, ValType, VecType, ImportDesc, GlobalType, Data, ByteVec, ExportDesc, BlockType,
}, memory_region::MemoryRegion};

#[derive(Debug)]
struct TKTK;

struct Stack {
    stack: Vec<StackItem>,
}

enum StackItem {
    Value,
    Label,
    Activation,
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
            (Self::F32(l0), Self::F32(r0)) => unsafe { std::mem::transmute::<&f32, &u32>(l0) == std::mem::transmute::<&f32, &u32>(r0) },
            (Self::F64(l0), Self::F64(r0)) => unsafe { std::mem::transmute::<&f64, &u64>(l0) == std::mem::transmute::<&f64, &u64>(r0) },
            (lhs, rhs) => lhs == rhs
        }
    }

    fn as_usize(&self) -> Option<usize> {
        Some(match self {
            Value::I32(xs) => *xs as usize,
            Value::I64(xs) => *xs as usize,
            _ => return None
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
            _ => return anyhow::bail!("expected i32 value")
        })
    }

    fn is_zero(&self) -> anyhow::Result<bool> {
        Ok(match self {
            Value::I32(xs) => *xs == 0,
            Value::I64(xs) => *xs == 0,
            Value::F32(xs) => *xs == 0.0,
            Value::F64(xs) => *xs == 0.0,
            _ => return anyhow::bail!("invalid conversion of value to int via is_zero")
        })
    }
}

#[derive(Debug)]
enum FuncInstImpl {
    Guest(FuncIdx),
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
    r#impl: TableInstImpl
}

#[derive(Debug)]
enum MemoryInstImpl {
    Guest(MemoryRegion),
    Host(TKTK),
}

#[derive(Debug)]
struct MemInst {
    r#type: MemType,
    r#impl: MemoryInstImpl
}

impl MemInst {
    fn copy_data(&mut self, byte_vec: &ByteVec<'_>, offset: usize) -> anyhow::Result<()> {
        match &mut self.r#impl {
            MemoryInstImpl::Guest(region) => {
                region.copy_data(byte_vec.0, offset)?
            },
            MemoryInstImpl::Host(_tktk) => todo!(),
        }

        Ok(())
    }

    #[inline]
    fn load<const U: usize>(&self, addr: usize) -> anyhow::Result<[u8; U]> {
        match &self.r#impl {
            MemoryInstImpl::Guest(memory) => {
                if addr.saturating_add(U) > memory.len() {
                    anyhow::bail!("out of bounds memory access")
                }
                Ok(
                    memory.as_slice()[addr..addr+U].try_into()?
                )
            },
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
    exports: HashMap<&'a str, (usize, ExportDesc)>,
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
    pub(crate) fn new(globals: Vec<Value>) -> Self { Self { globals } }
}

impl<'a> Module<'a> {
    pub(crate) fn new<'b: 'a>(module: ParsedModule<'b>) -> Self {
        Self {
            parsed_module: module,
        }
    }

    pub(crate) fn instantiate(&self, _imports: &Imports) -> anyhow::Result<ModuleInstance> {
        let (
            func_imports,
            memory_imports,
            table_imports,
            global_imports
        ) = self.parsed_module
            .import_section()
            .iter()
            .flat_map(|xs| xs.iter())
            .fold((vec![], vec![], vec![], vec![]), |(mut funcs, mut mems, mut tables, mut globals), imp| {
                match imp.desc {
                    ImportDesc::Func(desc) => funcs.push(desc),
                    ImportDesc::Mem(desc) => mems.push(desc),
                    ImportDesc::Table(desc) => tables.push(desc),
                    ImportDesc::Global(desc) => globals.push(desc),
                }

                (funcs, mems, tables, globals)
            });

        let globals: Vec<GlobalInst> = global_imports
            .into_iter()
            .map(|desc| {
                GlobalInst {
                    r#type: desc,
                    r#impl: GlobalInstImpl::Host(TKTK)
                }
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
                    Some(Instr::GlobalGet(LocalIdx(idx))) => {
                        let idx = *idx as usize;
                        globals[idx].value()
                    }
                    Some(Instr::RefNull(_ref_type)) => Value::RefNull,
                    Some(Instr::RefFunc(FuncIdx(idx))) => Value::RefFunc(FuncIdx(*idx)),
                    _ => anyhow::bail!("unsupported global initializer instruction"),
                };

                if !matches!(
                    (global_type.0, &global),
                    (ValType::NumType(NumType::I32), Value::I32(_))
                        | (ValType::NumType(NumType::I64), Value::I64(_))
                        | (ValType::NumType(NumType::I32), Value::F32(_))
                        | (ValType::NumType(NumType::F64), Value::F64(_))
                        | (ValType::VecType(VecType::V128), Value::V128(_))
                        | (ValType::RefType(RefType::FuncRef), Value::RefFunc(_))
                        | (ValType::RefType(RefType::FuncRef), Value::RefNull)
                        | (ValType::RefType(RefType::ExternRef), Value::RefExtern(_))
                ) {
                    anyhow::bail!("global type does not accept this value");
                }

                globals.push(GlobalInst {
                    r#type: *global_type,
                    r#impl: GlobalInstImpl::Guest(global)
                });
                Ok(globals)
            })?;

        // next step: build our function table.
        let functions = func_imports.iter()
            .map(|desc| {
                if desc.0 as usize >= self.parsed_module.type_section().map(|xs| xs.len()).unwrap_or_default() {
                    anyhow::bail!("type out of range");
                }

                Ok(FuncInst {
                    r#type: *desc,
                    r#impl: FuncInstImpl::Host(TKTK)
                })
            })
            .chain(
                self.parsed_module
                .function_section()
                .iter()
                .flat_map(|xs| xs.iter())
                .copied()
                .enumerate()
                .map(|(func_idx, xs)| {
                    if func_idx >= self.parsed_module.code_section().map(|xs| xs.len()).unwrap_or_default() {
                        anyhow::bail!("code idx out of range");
                    }

                    Ok(FuncInst {
                        r#type: xs,
                        r#impl: FuncInstImpl::Guest(FuncIdx(func_idx as u32))
                    })
                })
            )
        .collect::<anyhow::Result<Vec<_>>>()?;

        let mut memories: Vec<_> = memory_imports.iter()
            .map(|desc| {
                MemInst {
                    r#type: *desc,
                    r#impl: MemoryInstImpl::Host(TKTK)
                }
            }).chain(
                self.parsed_module
                .memory_section()
                .iter()
                .flat_map(|xs| xs.iter())
                .map(|memtype| {
                    MemInst {
                        r#type: *memtype,
                        r#impl: MemoryInstImpl::Guest(MemoryRegion::new(memtype.0))
                    }
                })
            )
            .collect();

        let tables: Vec<_> = table_imports.iter()
            .map(|desc| {
                TableInst {
                    r#type: *desc,
                    r#impl: TableInstImpl::Host(TKTK)
                }
            }).chain(
                self.parsed_module
                .table_section()
                .iter()
                .flat_map(|xs| xs.iter())
                .map(|tabletype| {
                    TableInst {
                        r#type: *tabletype,
                        r#impl: TableInstImpl::Guest(vec![Value::RefNull; tabletype.1.min() as usize])
                    }
                })
            )
            .collect();

        // TODO:
        // - [x] apply data to memories
        // - [ ] apply elems to tables
        for data in self.parsed_module
            .data_section()
            .iter()
            .flat_map(|xs| xs.iter()) {
            match data {
                Data::Active(data, memory_idx, expr) => {
                    let offset = compute_constant_expr(expr, globals.as_slice())?;
                    let offset = offset.as_usize().ok_or_else(|| anyhow::anyhow!("expected i32 or i64"))?;

                    memories[memory_idx.0 as usize].copy_data(data, offset)?;
                }
                Data::Passive(_) => continue
            }
        }

        let mut map = HashMap::new();
        for (idx, exports) in self.parsed_module
            .export_section()
            .iter()
            .flat_map(|xs| xs.iter())
            .enumerate() {

            map.insert(exports.nm.0, (idx, exports.desc));
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
        let Some((target_idx, ExportDesc::Func(type_idx))) = self.exports.get(funcname) else {
            anyhow::bail!("no such function, {}", funcname);
        };

        let Some(func) = self.functions.get(*target_idx) else {
            anyhow::bail!("no such function, {} (idx {} not in range)", funcname, target_idx);
        };

        let Some(typedef) = self.module.parsed_module.type_section().iter().flat_map(|xs| xs.iter()).nth(func.r#type.0 as usize) else {
            anyhow::bail!("no type definition for {}", funcname);
        };

        let param_types = typedef.0.0.as_slice();
        if args.len() < param_types.len() {
            anyhow::bail!("not enough arguments to call {}; expected {} args", funcname, param_types.len());
        }

        if args.len() > param_types.len() {
            anyhow::bail!("too many arguments to call {}; expected {} args", funcname, param_types.len());
        }

        for (idx, (param_type, value)) in param_types.iter().zip(args.iter()).enumerate() {
            match (param_type, value) {
                (ValType::NumType(NumType::I32), Value::I32(_)) |
                (ValType::NumType(NumType::I64), Value::I64(_)) |
                (ValType::NumType(NumType::F32), Value::F32(_)) |
                (ValType::NumType(NumType::F64), Value::F64(_)) |
                (ValType::VecType(VecType::V128), Value::V128(_)) |
                (ValType::RefType(RefType::FuncRef), Value::RefFunc(_)) |
                (ValType::RefType(RefType::FuncRef), Value::RefNull) |
                (ValType::RefType(RefType::ExternRef), Value::RefExtern(_)) |
                (ValType::RefType(RefType::ExternRef), Value::RefNull) => continue,
                (_vt, _v) => anyhow::bail!("bad argument at {}", idx)
            }
        }

        let FuncInstImpl::Guest(_func_inst_impl) = func.r#impl else {
            todo!("implement re-exported imports");
        };

        let Some(code) = self.module.parsed_module.code_section().iter().flat_map(|xs| xs.iter()).nth(*target_idx) else {
            anyhow::bail!("could not find code for function");
        };

        let locals = code.0.locals.as_slice();
        let mut instrs = code.0.expr.0.as_slice();

        let mut locals: Vec<Value> = args.iter().cloned().chain(
            locals.iter().map(|xs| {
                xs.1.instantiate()
            })
        ).collect();

        struct Frame<'a> {
            pc: usize,
            instrs: &'a [Instr],
            jump_to: usize
        }
        let mut stack = Vec::<Value>::new();
        let mut frames = Vec::<Frame<'a>>::new();
        frames.push(Frame { pc: 0, instrs, jump_to: instrs.len() });

        let mut frame_idx = 0;
        let mut pc = 0;

        loop {
            if pc >= instrs.len() {
                frames.pop();
                if frames.is_empty() {
                    break
                }
                frame_idx -= 1;
                instrs = frames[frame_idx].instrs;
                pc = frames[frame_idx].pc;
            }

            match &instrs[pc] {
                Instr::Unreachable => anyhow::bail!("hit trap"),
                Instr::Nop => {},
                Instr::Block(_ty, instrs) => {
                    frames[frame_idx].pc = pc;
                    frames.push(Frame {
                        pc: 0,
                        instrs,
                        jump_to: instrs.len(),
                    });
                },

                Instr::Loop(_ty, instrs) => {
                    frames[frame_idx].pc = pc;
                    frames.push(Frame {
                        pc: 0,
                        instrs,
                        jump_to: 0,
                    });
                },

                Instr::If(_, _) => todo!("If"),
                Instr::IfElse(_, _, _) => todo!("IfElse"),
                Instr::Br(idx) => {
                    if idx.0 > 0 {
                        frames.truncate(frames.len() - idx.0 as usize);
                        frame_idx = frames.len() - 1;
                        instrs = frames[frame_idx].instrs;
                    }
                    pc = frames[frame_idx].jump_to;
                },

                Instr::BrIf(idx) => {
                    let Some(v) = stack.pop() else {
                        anyhow::bail!("expected a value on the stack");
                    };

                    if !v.is_zero()? {
                        if idx.0 > 0 {
                            frames.truncate(frames.len() - idx.0 as usize);
                            frame_idx = frames.len() - 1;
                            instrs = frames[frame_idx].instrs;
                        }
                        pc = frames[frame_idx].jump_to;
                    }
                },
                Instr::BrTable(_, _) => todo!("BrTable"),
                Instr::Return => todo!("Return"),
                Instr::Call(_) => todo!("Call"),
                Instr::CallIndirect(_, _) => todo!("CallIndirect"),
                Instr::RefNull(_) => todo!("RefNull"),
                Instr::RefIsNull => todo!("RefIsNull"),
                Instr::RefFunc(_) => todo!("RefFunc"),
                Instr::Drop => todo!("Drop"),
                Instr::SelectEmpty => {
                    let items = stack.split_off(stack.len() - 3);
                    stack.push(if items[2].is_zero()? {
                        items[0]
                    } else {
                        items[1]
                    });
                },

                Instr::Select(_) => todo!("Select"),

                Instr::LocalGet(idx) => {
                    stack.push(locals[idx.0 as usize]);
                },

                Instr::LocalSet(idx) => {
                    locals[idx.0 as usize] = stack.pop().ok_or_else(|| anyhow::anyhow!("ran out of stack"))?;
                },

                Instr::LocalTee(idx) => {
                    locals[idx.0 as usize] = *stack.last().ok_or_else(|| anyhow::anyhow!("ran out of stack"))?;
                },

                Instr::GlobalGet(_) => todo!("GlobalGet"),
                Instr::GlobalSet(_) => todo!("GlobalSet"),
                Instr::TableGet(_) => todo!("TableGet"),
                Instr::TableSet(_) => todo!("TableSet"),
                Instr::TableInit(_, _) => todo!("TableInit"),
                Instr::ElemDrop(_) => todo!("ElemDrop"),
                Instr::TableCopy(_, _) => todo!("TableCopy"),
                Instr::TableGrow(_) => todo!("TableGrow"),
                Instr::TableSize(_) => todo!("TableSize"),
                Instr::TableFill(_) => todo!("TableFill"),

                Instr::I32Load(mem) => {
                    let v = stack.pop().ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = self.memories[mem.memidx()].load(offset)?;
                    stack.push(Value::I32(i32::from_le_bytes(arr)));
                },

                Instr::I64Load(mem) => {
                    let v = stack.pop().ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = self.memories[mem.memidx()].load(offset)?;
                    stack.push(Value::I64(i64::from_le_bytes(arr)));
                },

                Instr::F32Load(mem) => {
                    let v = stack.pop().ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = self.memories[mem.memidx()].load(offset)?;
                    stack.push(Value::F32(f32::from_le_bytes(arr)));
                },

                Instr::F64Load(mem) => {
                    let v = stack.pop().ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = self.memories[mem.memidx()].load(offset)?;
                    stack.push(Value::F64(f64::from_le_bytes(arr)));
                },

                Instr::I32Load8S(mem) => {
                    let v = stack.pop().ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = self.memories[mem.memidx()].load::<1>(offset)?;
                    stack.push(Value::I32(arr[0] as i32));
                },

                Instr::I32Load8U(mem) => {
                    let v = stack.pop().ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = self.memories[mem.memidx()].load::<1>(offset)?;
                    stack.push(Value::I32(arr[0] as u32 as i32));
                },

                Instr::I32Load16S(mem) => {
                    let v = stack.pop().ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = self.memories[mem.memidx()].load(offset)?;
                    stack.push(Value::I32(i16::from_le_bytes(arr) as i32));
                },

                Instr::I32Load16U(mem) => {
                    let v = stack.pop().ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = self.memories[mem.memidx()].load(offset)?;
                    stack.push(Value::I32(u16::from_le_bytes(arr) as i32));
                },

                Instr::I64Load8S(mem) => {
                    let v = stack.pop().ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = self.memories[mem.memidx()].load::<1>(offset)?;
                    stack.push(Value::I64(arr[0] as i64));
                },

                Instr::I64Load8U(mem) => {
                    let v = stack.pop().ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = self.memories[mem.memidx()].load::<1>(offset)?;
                    stack.push(Value::I64(arr[0] as u64 as i64));
                },

                Instr::I64Load16S(mem) => {
                    let v = stack.pop().ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = self.memories[mem.memidx()].load(offset)?;
                    stack.push(Value::I64(i16::from_le_bytes(arr) as i64));
                },

                Instr::I64Load16U(mem) => {
                    let v = stack.pop().ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = self.memories[mem.memidx()].load(offset)?;
                    stack.push(Value::I64(u16::from_le_bytes(arr) as i64));
                },

                Instr::I64Load32S(mem) => {
                    let v = stack.pop().ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = self.memories[mem.memidx()].load(offset)?;
                    stack.push(Value::I64(i32::from_le_bytes(arr) as i64));
                },

                Instr::I64Load32U(mem) => {
                    let v = stack.pop().ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = self.memories[mem.memidx()].load(offset)?;
                    stack.push(Value::I64(u32::from_le_bytes(arr) as i64));
                },

                Instr::I32Store(_) => todo!("I32Store"),
                Instr::I64Store(_) => todo!("I64Store"),
                Instr::F32Store(_) => todo!("F32Store"),
                Instr::F64Store(_) => todo!("F64Store"),
                Instr::I32Store8(_) => todo!("I32Store8"),
                Instr::I32Store16(_) => todo!("I32Store16"),
                Instr::I64Store8(_) => todo!("I64Store8"),
                Instr::I64Store16(_) => todo!("I64Store16"),
                Instr::I64Store32(_) => todo!("I64Store32"),
                Instr::MemorySize(_) => todo!("MemorySize"),
                Instr::MemoryGrow(_) => todo!("MemoryGrow"),
                Instr::MemoryInit(_, _) => todo!("MemoryInit"),
                Instr::DataDrop(_) => todo!("DataDrop"),
                Instr::MemoryCopy(_, _) => todo!("MemoryCopy"),
                Instr::MemoryFill(_) => todo!("MemoryFill"),
                Instr::I32Const(_) => todo!("I32Const"),
                Instr::I64Const(_) => todo!("I64Const"),
                Instr::F32Const(_) => todo!("F32Const"),
                Instr::F64Const(_) => todo!("F64Const"),
                Instr::I32Eqz => todo!("I32Eqz"),
                Instr::I32Eq => todo!("I32Eq"),
                Instr::I32Ne => todo!("I32Ne"),
                Instr::I32LtS => todo!("I32LtS"),
                Instr::I32LtU => todo!("I32LtU"),
                Instr::I32GtS => todo!("I32GtS"),
                Instr::I32GtU => todo!("I32GtU"),
                Instr::I32LeS => todo!("I32LeS"),
                Instr::I32LeU => todo!("I32LeU"),
                Instr::I32GeS => todo!("I32GeS"),
                Instr::I32GeU => todo!("I32GeU"),
                Instr::I64Eqz => todo!("I64Eqz"),
                Instr::I64Eq => todo!("I64Eq"),
                Instr::I64Ne => todo!("I64Ne"),
                Instr::I64LtS => todo!("I64LtS"),
                Instr::I64LtU => todo!("I64LtU"),
                Instr::I64GtS => todo!("I64GtS"),
                Instr::I64GtU => todo!("I64GtU"),
                Instr::I64LeS => todo!("I64LeS"),
                Instr::I64LeU => todo!("I64LeU"),
                Instr::I64GeS => todo!("I64GeS"),
                Instr::I64GeU => todo!("I64GeU"),
                Instr::F32Eq => todo!("F32Eq"),
                Instr::F32Ne => todo!("F32Ne"),
                Instr::F32Lt => todo!("F32Lt"),
                Instr::F32Gt => todo!("F32Gt"),
                Instr::F32Le => todo!("F32Le"),
                Instr::F32Ge => todo!("F32Ge"),
                Instr::F64Eq => todo!("F64Eq"),
                Instr::F64Ne => todo!("F64Ne"),
                Instr::F64Lt => todo!("F64Lt"),
                Instr::F64Gt => todo!("F64Gt"),
                Instr::F64Le => todo!("F64Le"),
                Instr::F64Ge => todo!("F64Ge"),
                Instr::I32Clz => todo!("I32Clz"),
                Instr::I32Ctz => todo!("I32Ctz"),
                Instr::I32Popcnt => todo!("I32Popcnt"),
                Instr::I32Add => {
                    let (Some(Value::I32(lhs)), Some(Value::I32(rhs))) = (stack.pop(), stack.pop()) else {
                        anyhow::bail!("failed to add");
                    };
                    stack.push(Value::I32(lhs + rhs));
                },
                Instr::I32Sub => todo!("I32Sub"),
                Instr::I32Mul => todo!("I32Mul"),
                Instr::I32DivS => todo!("I32DivS"),
                Instr::I32DivU => todo!("I32DivU"),
                Instr::I32RemS => todo!("I32RemS"),
                Instr::I32RemU => todo!("I32RemU"),
                Instr::I32And => todo!("I32And"),
                Instr::I32Ior => todo!("I32Ior"),
                Instr::I32Xor => todo!("I32Xor"),
                Instr::I32Shl => todo!("I32Shl"),
                Instr::I32ShrS => todo!("I32ShrS"),
                Instr::I32ShrU => todo!("I32ShrU"),
                Instr::I32Rol => todo!("I32Rol"),
                Instr::I32Ror => todo!("I32Ror"),
                Instr::I64Clz => todo!("I64Clz"),
                Instr::I64Ctz => todo!("I64Ctz"),
                Instr::I64Popcnt => todo!("I64Popcnt"),
                Instr::I64Add => todo!("I64Add"),
                Instr::I64Sub => todo!("I64Sub"),
                Instr::I64Mul => todo!("I64Mul"),
                Instr::I64DivS => todo!("I64DivS"),
                Instr::I64DivU => todo!("I64DivU"),
                Instr::I64RemS => todo!("I64RemS"),
                Instr::I64RemU => todo!("I64RemU"),
                Instr::I64And => todo!("I64And"),
                Instr::I64Ior => todo!("I64Ior"),
                Instr::I64Xor => todo!("I64Xor"),
                Instr::I64Shl => todo!("I64Shl"),
                Instr::I64ShrS => todo!("I64ShrS"),
                Instr::I64ShrU => todo!("I64ShrU"),
                Instr::I64Rol => todo!("I64Rol"),
                Instr::I64Ror => todo!("I64Ror"),
                Instr::F32Abs => todo!("F32Abs"),
                Instr::F32Neg => todo!("F32Neg"),
                Instr::F32Ceil => todo!("F32Ceil"),
                Instr::F32Floor => todo!("F32Floor"),
                Instr::F32Trunc => todo!("F32Trunc"),
                Instr::F32NearestInt => todo!("F32NearestInt"),
                Instr::F32Sqrt => todo!("F32Sqrt"),
                Instr::F32Add => todo!("F32Add"),
                Instr::F32Sub => todo!("F32Sub"),
                Instr::F32Mul => todo!("F32Mul"),
                Instr::F32Div => todo!("F32Div"),
                Instr::F32Min => todo!("F32Min"),
                Instr::F32Max => todo!("F32Max"),
                Instr::F32CopySign => todo!("F32CopySign"),
                Instr::F64Abs => todo!("F64Abs"),
                Instr::F64Neg => todo!("F64Neg"),
                Instr::F64Ceil => todo!("F64Ceil"),
                Instr::F64Floor => todo!("F64Floor"),
                Instr::F64Trunc => todo!("F64Trunc"),
                Instr::F64NearestInt => todo!("F64NearestInt"),
                Instr::F64Sqrt => todo!("F64Sqrt"),
                Instr::F64Add => todo!("F64Add"),
                Instr::F64Sub => todo!("F64Sub"),
                Instr::F64Mul => todo!("F64Mul"),
                Instr::F64Div => todo!("F64Div"),
                Instr::F64Min => todo!("F64Min"),
                Instr::F64Max => todo!("F64Max"),
                Instr::F64CopySign => todo!("F64CopySign"),
                Instr::I32ConvertI64 => todo!("I32ConvertI64"),
                Instr::I32SConvertF32 => todo!("I32SConvertF32"),
                Instr::I32UConvertF32 => todo!("I32UConvertF32"),
                Instr::I32SConvertF64 => todo!("I32SConvertF64"),
                Instr::I32UConvertF64 => todo!("I32UConvertF64"),
                Instr::I64SConvertI32 => todo!("I64SConvertI32"),
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
            pc += 1;
        }

        // TODO: handle multiple return values
        Ok(stack)
    }
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
            ValType::RefType(RefType::FuncRef) => Value::RefNull,
            ValType::RefType(RefType::ExternRef) => Value::RefNull,
            ValType::RefType(RefType::ExternRef) => Value::RefNull,
        }
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
        },

        Some(Instr::RefNull(_c)) => todo!(),
        Some(Instr::RefFunc(c)) => Value::RefFunc(*c),
        _ => anyhow::bail!("unsupported instruction"),
    })
}

#[cfg(test)]
mod test {
    use crate::parse::parse;
    use super::*;

    #[test]
    fn test_create_store() {
        let bytes = include_bytes!("../example2.wasm");

        let wasm = parse(bytes).unwrap();

        let module = Module::new(wasm);
        let imports = Imports::new(vec![]);
        let mut instance = module.instantiate(&imports).expect("could not instantiate module");
        let result = instance.call("add_i32", &[Value::I32(1), Value::I32(4)]).expect("math is hard");
        dbg!(result);
        // dbg!(xs);
    }
}
