use std::{any::Any, collections::HashMap, ops::Deref};

use anyhow::Context;

use crate::{
    memory_region::MemoryRegion,
    nodes::{
        BlockType, ByteVec, Code, CodeIdx, Data, Elem, ExportDesc, Expr, FuncIdx, Global, GlobalIdx, ImportDesc, Instr, MemIdx, Module, TableIdx, Type, TypeIdx, ValType
    },
};

use super::{
    function::FuncInst,
    global::{GlobalInst, GlobalInstImpl},
    imports::{GuestIndex, InternMap},
    memory::{MemInst, MemoryInstImpl},
    table::{TableInst, TableInstImpl},
    value::Value,
    Imports,
};

pub(super) type InstanceIdx = usize;
pub(super) type MachineCodeIndex = usize;
pub(super) type MachineMemoryIndex = usize;
pub(super) type MachineTableIndex = usize;
pub(super) type MachineGlobalIndex = usize;

impl Elem {
    pub(crate) fn value<'a>(&self, idx: usize, module_idx: usize, machine: &Machine<'a>) -> anyhow::Result<Value> {
        Ok(match self {
            Elem::ActiveSegmentFuncs(_, xs) |
            Elem::PassiveSegment(_, xs) |
            Elem::ActiveSegment(_, _, _, xs) |
            Elem::DeclarativeSegment(_, xs) => Value::RefFunc(xs[idx]),
            Elem::ActiveSegmentExpr(_, xs) | 
            Elem::PassiveSegmentExpr(_, xs) |
            Elem::ActiveSegmentTableAndExpr(_, _, _, xs) |
            Elem::DeclarativeSegmentExpr(_, xs) => machine.compute_constant_expr(module_idx, xs[idx].0.as_slice())?
        })
    }
}

#[derive(Debug)]
struct Frame<'a> {
    #[cfg(test)]
    name: &'static str,
    pc: usize,
    return_unwind_count: usize,
    instrs: &'a [Instr],
    jump_to: Option<usize>,
    block_type: BlockType,
    locals_base_offset: usize,
    value_stack_offset: usize,
    guest_index: GuestIndex,
}

pub(crate) struct Machine<'a> {
    initialized: bool,

    types: Box<[Box<[Type]>]>,
    code: Box<[Box<[Code]>]>,
    data: Box<[Box<[Data<'a>]>]>,
    elements: Box<[Box<[Elem]>]>,

    functions: Box<[Box<[FuncInst]>]>,
    globals: Box<[Box<[GlobalInst]>]>,
    memories: Box<[Box<[MemInst]>]>,
    tables: Box<[Box<[TableInst]>]>,

    memory_regions: Box<[MemoryRegion]>,
    global_values: Box<[Value]>,
    table_instances: Box<[Vec<Value>]>,
    exports: HashMap<usize, ExportDesc>,
    internmap: InternMap,
}

impl<'a> Machine<'a> {
    pub(super) fn new(
        mut imports: Imports<'a>,
        exports: HashMap<usize, ExportDesc>,
    ) -> anyhow::Result<Self> {
        let guests = imports.guests.split_off(0);

        let mut memory_regions: Vec<MemoryRegion> = Vec::new();
        let mut table_instances: Vec<Vec<Value>> = Vec::new();
        let mut global_values: Vec<Value> = Vec::new();

        let mut all_code: Vec<Box<[_]>> = Vec::with_capacity(guests.len());
        let mut all_data: Vec<Box<[_]>> = Vec::with_capacity(guests.len());
        let mut all_functions: Vec<Box<[_]>> = Vec::with_capacity(guests.len());
        let mut all_elements: Vec<Box<[_]>> = Vec::with_capacity(guests.len());
        let mut all_globals: Vec<Box<[_]>> = Vec::with_capacity(guests.len());
        let mut all_memories: Vec<Box<[_]>> = Vec::with_capacity(guests.len());
        let mut all_tables: Vec<Box<[_]>> = Vec::with_capacity(guests.len());
        let mut all_types: Vec<Box<[_]>> = Vec::with_capacity(guests.len());

        for guest in guests {
            let Module {
                type_section,
                function_section,
                table_section,
                import_section,
                memory_section,
                global_section,
                element_section,
                code_section,
                data_section,
                ..
            } = guest;

            let (
                type_section,
                function_section,
                import_section,
                table_section,
                memory_section,
                global_section,
                element_section,
                code_section,
                data_section,
            ) = (
                type_section.map(|xs| xs.inner).unwrap_or_default(),
                function_section.map(|xs| xs.inner).unwrap_or_default(),
                import_section.map(|xs| xs.inner).unwrap_or_default(),
                table_section.map(|xs| xs.inner).unwrap_or_default(),
                memory_section.map(|xs| xs.inner).unwrap_or_default(),
                global_section.map(|xs| xs.inner).unwrap_or_default(),
                element_section.map(|xs| xs.inner).unwrap_or_default(),
                code_section.map(|xs| xs.inner).unwrap_or_default(),
                data_section.map(|xs| xs.inner).unwrap_or_default(),
            );

            let (import_func_count, import_mem_count, import_table_count, import_global_count) =
                import_section
                    .iter()
                    .fold((0, 0, 0, 0), |(fc, mc, tc, gc), imp| match imp.desc {
                        ImportDesc::Func(_) => (fc + 1, mc, tc, gc),
                        ImportDesc::Mem(_) => (fc, mc + 1, tc, gc),
                        ImportDesc::Table(_) => (fc, mc, tc + 1, gc),
                        ImportDesc::Global(_) => (fc, mc, tc, gc + 1),
                    });

            let mut functions = Vec::with_capacity(function_section.len() + import_func_count);
            let mut globals = Vec::with_capacity(global_section.len() + import_global_count);
            let mut memories = Vec::with_capacity(memory_section.len() + import_mem_count);
            let mut tables = Vec::with_capacity(table_section.len() + import_table_count);
            for imp in import_section {
                match imp.desc {
                    ImportDesc::Func(desc) => {
                        functions.push(FuncInst::resolve(desc, &imp, &imports)?);
                    }
                    ImportDesc::Mem(desc) => {
                        memories.push(MemInst::resolve(desc, &imp, &imports)?);
                    }
                    ImportDesc::Table(desc) => {
                        tables.push(TableInst::resolve(desc, &imp, &imports)?);
                    }
                    ImportDesc::Global(desc) => {
                        globals.push(GlobalInst::resolve(desc, &imp, &imports)?);
                    }
                }
            }

            let mut saw_error = false;
            let code_max = code_section.len();
            functions.extend(
                function_section
                    .into_iter()
                    .enumerate()
                    .map(|(code_idx, xs)| {
                        if code_idx >= code_max {
                            saw_error = true;
                        }
                        FuncInst::new(xs, CodeIdx(code_idx as u32))
                    }),
            );

            if saw_error {
                anyhow::bail!("code idx out of range");
            }

            memories.extend(memory_section.into_iter().map(|memtype| {
                let memory_region_idx = memory_regions.len();
                memory_regions.push(MemoryRegion::new(memtype.0));
                MemInst::new(memtype, memory_region_idx)
            }));

            tables.extend(table_section.into_iter().map(|tabletype| {
                let table_instance_idx = table_instances.len();
                table_instances.push(vec![Value::RefNull; tabletype.1.min() as usize]);
                TableInst::new(tabletype, table_instance_idx)
            }));

            let global_base_offset = global_values.len();
            global_values.reserve(global_section.len());
            globals.extend(global_section.into_iter().enumerate().map(|(idx, global)| {
                let Global(global_type, Expr(instrs)) = global;
                global_values.push(global_type.0.instantiate());
                GlobalInst::new(
                    global_type,
                    global_base_offset + idx,
                    instrs.into_boxed_slice(),
                )
            }));

            all_code.push(code_section.into_boxed_slice());
            all_data.push(data_section.into_boxed_slice());
            all_elements.push(element_section.into_boxed_slice());
            all_functions.push(functions.into_boxed_slice());
            all_globals.push(globals.into_boxed_slice());
            all_memories.push(memories.into_boxed_slice());
            all_tables.push(tables.into_boxed_slice());
            all_types.push(type_section.into_boxed_slice());
        }

        Ok(Self {
            initialized: false,
            types: all_types.into_boxed_slice(),
            code: all_code.into_boxed_slice(),
            data: all_data.into_boxed_slice(),
            elements: all_elements.into_boxed_slice(),
            functions: all_functions.into_boxed_slice(),
            globals: all_globals.into_boxed_slice(),
            memories: all_memories.into_boxed_slice(),
            tables: all_tables.into_boxed_slice(),
            memory_regions: memory_regions.into_boxed_slice(),
            global_values: global_values.into_boxed_slice(),
            table_instances: table_instances.into_boxed_slice(),
            exports,
            internmap: imports.internmap,
        })
    }

    fn memory(&self, mut module_idx: usize, mut memory_idx: MemIdx) -> MachineMemoryIndex {
        loop {
            let memory = &self.memories[module_idx][memory_idx.0 as usize];
            match memory.r#impl {
                MemoryInstImpl::Local(machine_idx) => return machine_idx,
                MemoryInstImpl::Remote(midx, idx) => {
                    module_idx = midx;
                    memory_idx = idx;
                }
            }
        }
    }

    fn global(&self, mut module_idx: usize, mut global_idx: GlobalIdx) -> MachineGlobalIndex {
        loop {
            let global = &self.globals[module_idx][global_idx.0 as usize];
            match global.r#impl {
                GlobalInstImpl::Local(machine_idx, _) => return machine_idx,
                GlobalInstImpl::Remote(midx, idx) => {
                    module_idx = midx;
                    global_idx = idx;
                }
            }
        }
    }

    fn table(&self, mut module_idx: usize, mut table_idx: TableIdx) -> MachineTableIndex {
        loop {
            let table = &self.tables[module_idx][table_idx.0 as usize];
            match table.r#impl {
                TableInstImpl::Local(machine_idx) => return machine_idx,
                TableInstImpl::Remote(midx, idx) => {
                    module_idx = midx;
                    table_idx = idx;
                }
            }
        }
    }

    fn code(&self, module_idx: usize, code_idx: usize) -> Option<&Code> {
        self.code.get(module_idx)?.get(code_idx)
    }

    fn function(
        &self,
        mut module_idx: GuestIndex,
        mut func_idx: FuncIdx,
    ) -> Option<(GuestIndex, &FuncInst)> {
        loop {
            let func = self.functions.get(module_idx)?.get(func_idx.0 as usize);
            let Some(func) = func else {
                return None;
            };

            match func.r#impl {
                super::function::FuncInstImpl::Local(_) => return Some((module_idx, func)),
                super::function::FuncInstImpl::Remote(midx, fidx) => {
                    module_idx = midx;
                    func_idx = fidx;
                }
            }
        }
    }

    fn typedef(&self, module_idx: usize, type_idx: &TypeIdx) -> Option<&Type> {
        self.types.get(module_idx)?.get(type_idx.0 as usize)
    }

    pub(crate) fn initialize(&mut self) -> anyhow::Result<()> {
        for (module_idx, global_set) in self.globals.iter().enumerate() {
            for global in global_set.iter() {
                let Some((global_idx, instrs)) = global.initdata() else { continue };
                self.global_values[global_idx] = self.compute_constant_expr(module_idx, instrs)?;
            }
        }


        for (module_idx, data_set) in self.data.iter().enumerate() {
            for data in data_set.iter() {
                match data {
                    Data::Active(data, memory_idx, expr) => {
                        let memoffset = self.compute_constant_expr(module_idx, &expr.0)?;
                        let memoffset = memoffset
                            .as_usize()
                            .ok_or_else(|| anyhow::anyhow!("expected i32 or i64"))?;

                        let memoryidx = self.memory(module_idx, *memory_idx);
                        let memory = self.memory_regions.get_mut(memoryidx).ok_or_else(|| anyhow::anyhow!("no such memory"))?;

                        memory.grow_to_fit(data.0, memoffset)?;
                        memory.copy_data(data.0, memoffset);
                    }
                    Data::Passive(_) => continue,
                }
            }
        }

        for (module_idx, elem_set) in self.elements.iter().enumerate() {
            for elem in elem_set.iter() {
                match elem {
                    Elem::ActiveSegmentFuncs(expr, func_indices) => {
                        let offset = self.compute_constant_expr(module_idx, &expr.0)?;
                        let offset = offset
                            .as_usize()
                            .ok_or_else(|| anyhow::anyhow!("expected i32 or i64"))?;

                        let tableidx = self.table(module_idx, TableIdx(0));
                        let table = self.table_instances.get_mut(tableidx).ok_or_else(|| anyhow::anyhow!("no such table"))?;

                        for (idx, xs) in func_indices.iter().enumerate() {
                            table[idx + offset] = Value::RefFunc(*xs);
                        }
                    }

                    // "elemkind" means "funcref" or "externref"
                    Elem::ActiveSegment(table_idx, expr, _elemkind, func_indices) => {
                        let offset = self.compute_constant_expr(module_idx, &expr.0)?;
                        let offset = offset
                            .as_usize()
                            .ok_or_else(|| anyhow::anyhow!("expected i32 or i64"))?;

                        let tableidx = self.table(module_idx, *table_idx);
                        let table = self.table_instances.get_mut(tableidx).ok_or_else(|| anyhow::anyhow!("no such table"))?;

                        for (idx, xs) in func_indices.iter().enumerate() {
                            table[idx + offset] = Value::RefFunc(*xs);
                        }
                    },

                    Elem::ActiveSegmentExpr(_, _) => todo!("ActiveSegmentExpr"),
                    Elem::ActiveSegmentTableAndExpr(_, _, _, _) => todo!("ActiveSegmentTableAndExpr"),

                    _ => (), // passive and declarative segments do not participate in
                             // initialization
                }
            }
        }

        self.initialized = true;
        Ok(())
    }

    pub(crate) fn call(&mut self, funcname: &str, args: &[Value]) -> anyhow::Result<Vec<Value>> {
        if !self.initialized {
            self.initialize()?;
        }

        // how to get lookup funcname? (have to preserve the last module's externs)
        let func_name_idx = self
            .internmap
            .get(funcname)
            .ok_or_else(|| anyhow::anyhow!("no such export {funcname}"))?;

        let ExportDesc::Func(func_idx) = self.exports.get(&func_name_idx)
            .ok_or_else(|| anyhow::anyhow!("no such export {funcname}"))? else {
                anyhow::bail!("export {funcname} is not a function");
            };

        let module_idx = self.functions.len() - 1;
        let (module_idx, function) = self
            .function(module_idx, *func_idx)
            .ok_or_else(|| anyhow::anyhow!("missing final module"))?;
        let typedef = self
            .typedef(module_idx, function.typeidx())
            .ok_or_else(|| anyhow::anyhow!("missing typedef"))?;

        let param_types = typedef.0 .0.as_slice();
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
            param_type
                .validate(value)
                .with_context(|| format!("bad argument at {}", idx))?;
        }
        let code = &self.code[module_idx][function.codeidx().0 as usize];
        let locals = code.0.locals.as_slice();

        let mut locals: Vec<Value> = args
            .iter()
            .cloned()
            .chain(locals.iter().flat_map(|xs| (0..xs.0).map(|_| xs.1.instantiate())))
            .collect();

        let mut value_stack = Vec::<Value>::new();
        let mut frames = Vec::<Frame<'a>>::new();
        frames.push(Frame {
            #[cfg(test)]
            name: "init",
            pc: 0,
            return_unwind_count: 0,
            instrs: code.0.expr.0.as_slice(),
            jump_to: Some(code.0.expr.0.len()),
            locals_base_offset: 0,
            block_type: BlockType::TypeIndex(*function.typeidx()),
            value_stack_offset: 0,
            guest_index: module_idx,
        });

        // eprintln!("= = = = {funcname} {code:?}");
        loop {
            //eprintln!("value_stack={value_stack:?}; frame_idx={}@{}/{} -> {:?}", frames.len()- 1,frames[frames.len() -1].pc,frames[frames.len() -1].instrs.len(), &frames[frames.len() -1].instrs[frames[frames.len() -1].pc..]);
            let frame_idx = frames.len() - 1;
            if frames[frame_idx].pc >= frames[frame_idx].instrs.len() {
                locals.shrink_to(frames[frame_idx].locals_base_offset);
                let _last_frame = frames
                    .pop()
                    .expect("we should always be able to pop a frame");

                if frames.is_empty() {
                    break;
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
                        value_stack_offset: value_stack.len(),
                        guest_index: frames[frame_idx].guest_index,
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
                        value_stack_offset: value_stack.len(),
                        guest_index: frames[frame_idx].guest_index,
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
                        consequent.deref()
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
                        value_stack_offset: value_stack.len(),
                        guest_index: frames[frame_idx].guest_index,
                    });
                }

                Instr::IfElse(block_type, consequent, alternate) => {
                    if value_stack.is_empty() {
                        anyhow::bail!("expected 1 value on stack");
                    }

                    let blockinstrs = if let Some(Value::I32(0) | Value::I64(0)) = value_stack.pop()
                    {
                        alternate.deref()
                    } else {
                        consequent.deref()
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
                        value_stack_offset: value_stack.len(),
                        guest_index: frames[frame_idx].guest_index,
                    });
                }

                Instr::Br(idx) => {
                    // look at the arity of the target block type. preserve that many values from
                    // the stack.
                    if idx.0 > 0 {
                        frames.truncate(frames.len() - idx.0 as usize);
                    }

                    let frame_idx = frames.len() - 1;
                    let Some(jump_to) = frames[frame_idx].jump_to else {
                        anyhow::bail!("invalid jump target");
                    };
                    frames[frame_idx].pc = jump_to;
                    let to_preserve = match frames[frame_idx].block_type {
                        BlockType::Val(val_type) => {
                            if value_stack.is_empty() {
                                anyhow::bail!("block expected at least one value on stack");
                            }
                            let vs = value_stack.split_off(value_stack.len() - 1);
                            val_type.validate(&vs[0])?;
                            vs
                        },
                        BlockType::Empty => vec![],
                        BlockType::TypeIndex(type_idx) => {
                            let Some(ty) = self.types[frames[frame_idx].guest_index].get(type_idx.0 as usize) else {
                                anyhow::bail!("could not resolve blocktype");
                            };
                            if value_stack.len() < ty.1.0.len() {
                                anyhow::bail!("block expected at least {} value{} on stack", ty.1.0.len(), if ty.1.0.len() == 1 { "" } else { "s" });
                            }
                            let vs = value_stack.split_off(value_stack.len() - ty.1.0.len());
                            if let Some(err) = ty.1.0.iter().enumerate().map(|(idx, vt)| vt.validate(&vs[idx])).find(|xs| { xs.is_err() }) {
                                err?
                            };
                            vs
                        },
                    };
                    value_stack.truncate(frames[frame_idx].value_stack_offset);
                    value_stack.extend_from_slice(to_preserve.as_slice());

                    continue;
                }

                Instr::BrIf(idx) => {
                    let Some(v) = value_stack.pop() else {
                        anyhow::bail!("br.if: expected a value on the stack");
                    };

                    if !v.is_zero()? {
                        if idx.0 > 0 {
                            frames.truncate(frames.len() - idx.0 as usize);
                        }

                        let frame_idx = frames.len() - 1;
                        let Some(jump_to) = frames[frame_idx].jump_to else {
                            anyhow::bail!("invalid jump target");
                        };
                        frames[frame_idx].pc = jump_to;

                        let to_preserve = match frames[frame_idx].block_type {
                            BlockType::Val(val_type) => {
                                if value_stack.is_empty() {
                                    anyhow::bail!("block expected at least one value on stack");
                                }
                                let vs = value_stack.split_off(value_stack.len() - 1);
                                val_type.validate(&vs[0])?;
                                vs
                            },
                            BlockType::Empty => vec![],
                            BlockType::TypeIndex(type_idx) => {
                                let Some(ty) = self.types[frames[frame_idx].guest_index].get(type_idx.0 as usize) else {
                                    anyhow::bail!("could not resolve blocktype");
                                };
                                if value_stack.len() < ty.1.0.len() {
                                    anyhow::bail!("block expected at least {} value{} on stack", ty.1.0.len(), if ty.1.0.len() == 1 { "" } else { "s" });
                                }
                                let vs = value_stack.split_off(value_stack.len() - ty.1.0.len());
                                if let Some(err) = ty.1.0.iter().enumerate().map(|(idx, vt)| vt.validate(&vs[idx])).find(|xs| { xs.is_err() }) {
                                    err?
                                };
                                vs
                            },
                        };
                        value_stack.truncate(frames[frame_idx].value_stack_offset);
                        value_stack.extend_from_slice(to_preserve.as_slice());
                        continue;
                    }
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
                    let to_preserve = match frames[frame_idx].block_type {
                        BlockType::Val(val_type) => {
                            if value_stack.is_empty() {
                                anyhow::bail!("block expected at least one value on stack");
                            }
                            let vs = value_stack.split_off(value_stack.len() - 1);
                            val_type.validate(&vs[0])?;
                            vs
                        },
                        BlockType::Empty => vec![],
                        BlockType::TypeIndex(type_idx) => {
                            let Some(ty) = self.types[frames[frame_idx].guest_index].get(type_idx.0 as usize) else {
                                anyhow::bail!("could not resolve blocktype");
                            };
                            if value_stack.len() < ty.1.0.len() {
                                anyhow::bail!("block expected at least {} value{} on stack", ty.1.0.len(), if ty.1.0.len() == 1 { "" } else { "s" });
                            }
                            let vs = value_stack.split_off(value_stack.len() - ty.1.0.len());
                            if let Some(err) = ty.1.0.iter().enumerate().map(|(idx, vt)| vt.validate(&vs[idx])).find(|xs| { xs.is_err() }) {
                                err?
                            };
                            vs
                        },
                    };
                    value_stack.truncate(frames[frame_idx].value_stack_offset);
                    value_stack.extend_from_slice(to_preserve.as_slice());
                    continue;
                }

                Instr::Return => {
                    // TODO: validate result type!
                    let ruc = frames[frame_idx].return_unwind_count;
                    if ruc > 0 {
                        frames.truncate(frames.len() - ruc);
                    }

                    if frames.is_empty() {
                        break;
                    }

                    let new_frame_idx = frames.len() - 1;
                    frames[new_frame_idx].pc = frames[new_frame_idx].instrs.len();
                    let to_preserve = match frames[new_frame_idx].block_type {
                        BlockType::Val(val_type) => {
                            if value_stack.is_empty() {
                                anyhow::bail!("block expected at least one value on stack");
                            }
                            let vs = value_stack.split_off(value_stack.len() - 1);
                            val_type.validate(&vs[0])?;
                            vs
                        },
                        BlockType::Empty => vec![],
                        BlockType::TypeIndex(type_idx) => {
                            let Some(ty) = self.types[frames[new_frame_idx].guest_index].get(type_idx.0 as usize) else {
                                anyhow::bail!("could not resolve blocktype");
                            };
                            if value_stack.len() < ty.1.0.len() {
                                anyhow::bail!("block expected at least {} value{} on stack", ty.1.0.len(), if ty.1.0.len() == 1 { "" } else { "s" });
                            }
                            let vs = value_stack.split_off(value_stack.len() - ty.1.0.len());
                            if let Some(err) = ty.1.0.iter().enumerate().map(|(idx, vt)| vt.validate(&vs[idx])).find(|xs| { xs.is_err() }) {
                                err?
                            };
                            vs
                        },
                    };
                    value_stack.truncate(frames[new_frame_idx].value_stack_offset);
                    value_stack.extend_from_slice(to_preserve.as_slice());
                    continue;
                }

                Instr::Call(func_idx) => {
                    let (module_idx, function) = self
                        .function(module_idx, *func_idx)
                        .ok_or_else(|| anyhow::anyhow!("missing function"))?;

                    let typedef = self
                        .typedef(module_idx, function.typeidx())
                        .ok_or_else(|| anyhow::anyhow!("missing typedef"))?;

                    let code = &self.code[module_idx][function.codeidx().0 as usize];

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
                    let locals_count = code.0.locals.iter().fold(0, |lhs, rhs| lhs + rhs.0);
                    locals.reserve(locals_count as usize);

                    for (idx, (param_type, value)) in
                        param_types.iter().zip(args.iter()).enumerate()
                    {
                        param_type
                            .validate(value)
                            .with_context(|| format!("bad argument at {}", idx))?;
                        locals.push(*value);
                    }

                    for local in code.0.locals.iter().skip(param_types.len()) {
                        for _ in 0..local.0 {
                            locals.push(local.1.instantiate());
                        }
                    }

                    frames.push(Frame {
                        #[cfg(test)]
                        name: "Call",
                        pc: 0,
                        return_unwind_count: 0,
                        instrs: &code.0.expr.0,
                        jump_to: None,
                        locals_base_offset,
                        block_type: BlockType::TypeIndex(*function.typeidx()),
                        value_stack_offset: value_stack.len(),
                        guest_index: module_idx,
                    });
                }

                Instr::CallIndirect(type_idx, table_idx) => {
                    let table_instance_idx = self.table(frames[frame_idx].guest_index, *table_idx);
                    let check_type = &self.types[frames[frame_idx].guest_index][type_idx.0 as usize];

                    let table = &self.table_instances[table_instance_idx];

                    let Some(Value::I32(idx)) = value_stack.pop() else {
                        anyhow::bail!("expected an i32 value on the stack");
                    };

                    // let TableType(_reftype, _limits) = &table.r#type;

                    let Some(v) = table.get(idx as usize) else {
                        anyhow::bail!("undefined element: table index out of range");
                    };

                    if let Value::RefNull = &v {
                        anyhow::bail!("uninitialized element {idx}");
                    };

                    let Value::RefFunc(v) = v else {
                        anyhow::bail!("expected reffunc value, got {:?}", v);
                    };

                    let (module_idx, function) = self
                        .function(module_idx, *v)
                        .ok_or_else(|| anyhow::anyhow!("missing function"))?;

                    let typedef = self
                        .typedef(module_idx, function.typeidx())
                        .ok_or_else(|| anyhow::anyhow!("missing typedef"))?;

                    if check_type != typedef {
                        anyhow::bail!("indirect call type mismatch");
                    }

                    let code = &self.code[module_idx][function.codeidx().0 as usize];

                    let param_types = typedef.0 .0.as_slice();
                    if value_stack.len() < param_types.len() {
                        anyhow::bail!(
                            "not enough arguments to call func idx={}; expected {} args",
                            v.0,
                            param_types.len()
                        );
                    }

                    let args = value_stack.split_off(value_stack.len() - param_types.len());

                    let locals_base_offset = locals.len();
                    let locals_count = code.0.locals.iter().fold(0, |lhs, rhs| lhs + rhs.0);
                    locals.reserve(locals_count as usize);

                    for (idx, (param_type, value)) in
                        param_types.iter().zip(args.iter()).enumerate()
                    {
                        param_type
                            .validate(value)
                            .with_context(|| format!("bad argument at {}", idx))?;
                        locals.push(*value);
                    }

                    for local in code.0.locals.iter().skip(param_types.len()) {
                        for _ in 0..local.0 {
                            locals.push(local.1.instantiate());
                        }
                    }

                    frames.push(Frame {
                        #[cfg(test)]
                        name: "CallIndirect",
                        pc: 0,
                        return_unwind_count: 0,
                        instrs: &code.0.expr.0,
                        jump_to: None,
                        locals_base_offset,
                        block_type: BlockType::TypeIndex(*function.typeidx()),
                        value_stack_offset: value_stack.len(),
                        guest_index: module_idx,
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
                        anyhow::bail!(
                            "local.get out of range {} + {} > {}",
                            idx.0 as usize,
                            frames[frame_idx].locals_base_offset,
                            locals.len()
                        )
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
                    let global_value_idx = self.global(frames[frame_idx].guest_index, *global_idx);

                    // TODO: respect base globals offset
                    let Some(value) = self.global_values.get(global_value_idx) else {
                        anyhow::bail!("global idx out of range");
                    };

                    value_stack.push(*value);
                }

                Instr::GlobalSet(global_idx) => {
                    let global_value_idx = self.global(frames[frame_idx].guest_index, *global_idx);
                    let Some(value) = self.global_values.get_mut(global_value_idx) else {
                        anyhow::bail!("global idx out of range");
                    };

                    let Some(v) = value_stack.pop() else {
                        anyhow::bail!("drop out of range")
                    };

                    *value = v;
                }

                Instr::TableGet(_) => todo!("TableGet"),
                Instr::TableSet(_) => todo!("TableSet"),

                Instr::TableInit(elem_idx, table_idx) => {
                    let guest_index = frames[frame_idx].guest_index;
                    let Some(elem) = self.elements.get(guest_index).and_then(|xs| xs.get(elem_idx.0 as usize)) else {
                        anyhow::bail!("element idx out of range");
                    };

                    let table_idx = self.table(frames[frame_idx].guest_index, *table_idx);

                    let items = value_stack.split_off(value_stack.len() - 3);
                    let Some(count) = items[2].as_usize() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };
                    let Some(srcaddr) = items[1].as_usize() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };
                    let Some(destaddr) = items[0].as_usize() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };

                    let elem_len = elem.len();
                    if (count + srcaddr) > elem_len {
                        anyhow::bail!("out of bounds table access");
                    }
                    if srcaddr > elem_len {
                        anyhow::bail!("out of bounds table access");
                    }

                    let v: Box<[Value]> = (srcaddr..srcaddr + count).map(|idx| {
                        elem.value(idx, guest_index, self)
                    }).collect::<anyhow::Result<_>>()?;

                    let table = &mut self.table_instances[table_idx];
                    if (destaddr + count) > table.len() {
                        anyhow::bail!("out of bounds table access");
                    }
                    if destaddr > table.len() {
                        anyhow::bail!("out of bounds table access");
                    }

                    for (src_idx, dst_idx) in (0..count).zip(destaddr..destaddr+count) {
                        table[dst_idx] = v[src_idx];
                    }
                },

                Instr::ElemDrop(elem_idx) => {
                    let guest_index = frames[frame_idx].guest_index;
                    let Some(elem) = self.elements.get_mut(guest_index).and_then(|xs| xs.get_mut(elem_idx.0 as usize)) else {
                        anyhow::bail!("element idx out of range");
                    };

                    *elem = Elem::PassiveSegment(0, vec![]);
                },

                Instr::TableCopy(from_table_idx, to_table_idx) => {
                    let from_table_idx =
                        self.table(frames[frame_idx].guest_index, *from_table_idx);

                    let to_table_idx =
                        self.table(frames[frame_idx].guest_index, *to_table_idx);

                    let items = value_stack.split_off(value_stack.len() - 3);
                    let Some(count) = items[2].as_usize() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };
                    let Some(srcaddr) = items[1].as_usize() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };
                    let Some(destaddr) = items[0].as_usize() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };
                    if srcaddr > self.table_instances[from_table_idx].len() {
                        anyhow::bail!("out of bounds table access");
                    }
                    if destaddr > self.table_instances[to_table_idx].len() {
                        anyhow::bail!("out of bounds table access");
                    }

                    let values = (srcaddr..srcaddr+count).map(|src_idx| {
                        self.table_instances[from_table_idx][src_idx]
                    }).collect::<Box<[_]>>();

                    for (src_idx, dst_idx) in (0..count).zip(destaddr..destaddr+count) {
                        self.table_instances[to_table_idx][dst_idx] = values[src_idx];
                    }
                },

                Instr::TableGrow(_) => todo!("TableGrow"),
                Instr::TableSize(_) => todo!("TableSize"),
                Instr::TableFill(_) => todo!("TableFill"),

                Instr::I32Load(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &self.memory_regions[memory_idx];

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = memory_region.load(offset)?;
                    value_stack.push(Value::I32(i32::from_le_bytes(arr)));
                }

                Instr::I64Load(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &self.memory_regions[memory_idx];

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = memory_region.load(offset)?;
                    value_stack.push(Value::I64(i64::from_le_bytes(arr)));
                }

                Instr::F32Load(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &self.memory_regions[memory_idx];

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = memory_region.load(offset)?;
                    value_stack.push(Value::F32(f32::from_le_bytes(arr)));
                }

                Instr::F64Load(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &self.memory_regions[memory_idx];

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = memory_region.load(offset)?;
                    value_stack.push(Value::F64(f64::from_le_bytes(arr)));
                }

                Instr::I32Load8S(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &self.memory_regions[memory_idx];

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = memory_region.load::<1>(offset)?;
                    value_stack.push(Value::I32(arr[0] as i32));
                }

                Instr::I32Load8U(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &self.memory_regions[memory_idx];

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = memory_region.load::<1>(offset)?;
                    value_stack.push(Value::I32(arr[0] as u32 as i32));
                }

                Instr::I32Load16S(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &self.memory_regions[memory_idx];

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = memory_region.load(offset)?;
                    value_stack.push(Value::I32(i16::from_le_bytes(arr) as i32));
                }

                Instr::I32Load16U(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &self.memory_regions[memory_idx];

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = memory_region.load(offset)?;
                    value_stack.push(Value::I32(u16::from_le_bytes(arr) as i32));
                }

                Instr::I64Load8S(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &self.memory_regions[memory_idx];

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = memory_region.load::<1>(offset)?;
                    value_stack.push(Value::I64(arr[0] as i64));
                }

                Instr::I64Load8U(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &self.memory_regions[memory_idx];

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = memory_region.load::<1>(offset)?;
                    value_stack.push(Value::I64(arr[0] as u64 as i64));
                }

                Instr::I64Load16S(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &self.memory_regions[memory_idx];

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = memory_region.load(offset)?;
                    value_stack.push(Value::I64(i16::from_le_bytes(arr) as i64));
                }

                Instr::I64Load16U(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &self.memory_regions[memory_idx];

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = memory_region.load(offset)?;
                    value_stack.push(Value::I64(u16::from_le_bytes(arr) as i64));
                }

                Instr::I64Load32S(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &self.memory_regions[memory_idx];

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = memory_region.load(offset)?;
                    value_stack.push(Value::I64(i32::from_le_bytes(arr) as i64));
                }

                Instr::I64Load32U(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &self.memory_regions[memory_idx];

                    let v = value_stack
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("expected a value on the stack"))?;
                    let offset = mem.offset().saturating_add(v.as_mem_offset()?);
                    let arr = memory_region.load(offset)?;
                    value_stack.push(Value::I64(u32::from_le_bytes(arr) as i64));
                }

                Instr::I32Store(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &mut self.memory_regions[memory_idx];

                    let Some(Value::I32(v)) = value_stack.pop() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };

                    let Some(Value::I32(t)) = value_stack.pop() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };

                    let offset = mem.offset().saturating_add(t as usize);
                    memory_region.store(offset, &v.to_le_bytes())?;
                }

                Instr::I64Store(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &mut self.memory_regions[memory_idx];

                    let Some(Value::I64(v)) = value_stack.pop() else {
                        anyhow::bail!("expected i64 value on the stack")
                    };

                    let Some(Value::I32(t)) = value_stack.pop() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };

                    let offset = mem.offset().saturating_add(t as usize);
                    memory_region.store(offset, &v.to_le_bytes())?;
                },

                Instr::F32Store(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &mut self.memory_regions[memory_idx];

                    let Some(Value::F32(v)) = value_stack.pop() else {
                        anyhow::bail!("expected f32 value on the stack")
                    };

                    let Some(Value::I32(t)) = value_stack.pop() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };

                    let offset = mem.offset().saturating_add(t as usize);
                    memory_region.store(offset, &v.to_le_bytes())?;
                },

                Instr::F64Store(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &mut self.memory_regions[memory_idx];

                    let Some(Value::F64(v)) = value_stack.pop() else {
                        anyhow::bail!("expected f64 value on the stack")
                    };

                    let Some(Value::I32(t)) = value_stack.pop() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };

                    let offset = mem.offset().saturating_add(t as usize);
                    memory_region.store(offset, &v.to_le_bytes())?;
                },
                Instr::I32Store8(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &mut self.memory_regions[memory_idx];

                    let Some(Value::I32(v)) = value_stack.pop() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };

                    let Some(Value::I32(t)) = value_stack.pop() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };

                    let v = v as i8;
                    let offset = mem.offset().saturating_add(t as usize);
                    memory_region.store(offset, &[v as u8])?;
                },
                Instr::I32Store16(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &mut self.memory_regions[memory_idx];

                    let Some(Value::I32(v)) = value_stack.pop() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };

                    let Some(Value::I32(t)) = value_stack.pop() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };

                    let v = v as i16;
                    let offset = mem.offset().saturating_add(t as usize);
                    memory_region.store(offset, &v.to_le_bytes())?;
                },

                Instr::I64Store8(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &mut self.memory_regions[memory_idx];

                    let Some(Value::I64(v)) = value_stack.pop() else {
                        anyhow::bail!("expected i64 value on the stack")
                    };

                    let Some(Value::I32(t)) = value_stack.pop() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };

                    let v = v as i8;
                    let offset = mem.offset().saturating_add(t as usize);
                    memory_region.store(offset, &[v as u8])?;
                },
                Instr::I64Store16(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &mut self.memory_regions[memory_idx];

                    let Some(Value::I64(v)) = value_stack.pop() else {
                        anyhow::bail!("expected i64 value on the stack")
                    };

                    let Some(Value::I32(t)) = value_stack.pop() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };

                    let v = v as i16;
                    let offset = mem.offset().saturating_add(t as usize);
                    memory_region.store(offset, &v.to_le_bytes())?;
                },
                Instr::I64Store32(mem) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &mut self.memory_regions[memory_idx];

                    let Some(Value::I64(v)) = value_stack.pop() else {
                        anyhow::bail!("expected i64 value on the stack")
                    };

                    let Some(Value::I32(t)) = value_stack.pop() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };

                    let v = v as i32;
                    let offset = mem.offset().saturating_add(t as usize);
                    memory_region.store(offset, &v.to_le_bytes())?;
                },

                Instr::MemorySize(_) => todo!("MemorySize"),
                Instr::MemoryGrow(mem_idx) => {
                    let memory_idx = self.memory(frames[frame_idx].guest_index, *mem_idx);
                    let memory_region = &mut self.memory_regions[memory_idx];

                    let Some(Value::I32(v)) = value_stack.pop() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };

                    let page_count = memory_region.grow(v as usize)?;
                    value_stack.push(Value::I32(page_count as i32));
                }
                Instr::MemoryInit(data_idx, mem_idx) => {
                    let Some(data) = self.data[frames[frame_idx].guest_index].get(data_idx.0 as usize) else {
                        anyhow::bail!("could not fetch data by that id")
                    };
                    let data = match data {
                        Data::Active(v, _, _) => v,
                        Data::Passive(v) => v,
                    };

                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, *mem_idx);

                    let memory_region = &mut self.memory_regions[memory_idx];
                    let items = value_stack.split_off(value_stack.len() - 3);
                    let Some(count) = items[2].as_usize() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };
                    let Some(srcaddr) = items[1].as_usize() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };
                    let Some(destaddr) = items[0].as_usize() else {
                        anyhow::bail!("expected i32 value on the stack")
                    };

                    if srcaddr > data.0.len() {
                        anyhow::bail!("out of bounds memory access");
                    }
                    if (srcaddr + count) > data.0.len() {
                        anyhow::bail!("out of bounds memory access");
                    }
                    let from_slice = &data.0[srcaddr..(srcaddr + count)];

                    if destaddr > memory_region.len() {
                        anyhow::bail!("out of bounds memory access");
                    }

                    if (destaddr + count) > memory_region.len() {
                        anyhow::bail!("out of bounds memory access");
                    }

                    memory_region.copy_data(from_slice, destaddr);
                },

                Instr::DataDrop(data_idx) => {
                    let Some(data) = self.data[frames[frame_idx].guest_index].get_mut(data_idx.0 as usize) else {
                        anyhow::bail!("could not fetch data by that id")
                    };

                    *data = Data::Passive(ByteVec(&[]));
                },

                Instr::MemoryCopy(from_mem_idx, to_mem_idx) => {
                    let from_memory_idx =
                        self.memory(frames[frame_idx].guest_index, *from_mem_idx);

                    let to_memory_idx =
                        self.memory(frames[frame_idx].guest_index, *to_mem_idx);

                    // if these are the same memory, we're going to borrow it once mutably.
                    if from_memory_idx == to_memory_idx {
                        let memory_region = &mut self.memory_regions[from_memory_idx];
                        let items = value_stack.split_off(value_stack.len() - 3);
                        let Some(count) = items[2].as_usize() else {
                            anyhow::bail!("expected i32 value on the stack")
                        };
                        let Some(srcaddr) = items[1].as_usize() else {
                            anyhow::bail!("expected i32 value on the stack")
                        };
                        let Some(destaddr) = items[0].as_usize() else {
                            anyhow::bail!("expected i32 value on the stack")
                        };
                        if srcaddr > memory_region.len() {
                            anyhow::bail!("out of bounds memory access");
                        }
                        let from_slice = &memory_region.as_slice()[srcaddr..srcaddr+count];
                        if destaddr > memory_region.len() {
                            anyhow::bail!("out of bounds memory access");
                        }
                        memory_region.copy_overlapping_data(from_slice, destaddr);
                    } else {
                        let from_memory_region = &self.memory_regions[from_memory_idx];
                        let to_memory_region = &self.memory_regions[to_memory_idx];
                        // memory_region.copy_data(from_slice, count);
                        todo!()
                    }

                },

                Instr::MemoryFill(mem_idx) => {
                    let memory_idx =
                        self.memory(frames[frame_idx].guest_index, *mem_idx);

                    let memory_region = &mut self.memory_regions[memory_idx];

                    let items = value_stack.split_off(value_stack.len() - 3);
                    let Value::I32(count) = items[2] else {
                        anyhow::bail!("expected i32 value on the stack")
                    };
                    let Value::I32(val) = items[1] else {
                        anyhow::bail!("expected i32 value on the stack")
                    };
                    let Value::I32(offset) = items[0] else {
                        anyhow::bail!("expected i32 value on the stack")
                    };

                    memory_region.fill_data(val as u8, offset as usize, count as usize)?;
                },

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
                    let Some(v) = value_stack.pop() else {
                        anyhow::bail!("i32.eqz: expected 1 value on stack");
                    };
                    value_stack.push(Value::I32(if v.is_zero()? { 1 } else { 0 }));
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
                    let Some(v) = value_stack.pop() else {
                        anyhow::bail!("i64.eqz: expected 1 value on stack");
                    };
                    value_stack.push(Value::I64(if v.is_zero()? { 1 } else { 0 }));
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

                Instr::I32ConvertI64 => {
                    let Some(Value::I64(op)) = value_stack.pop() else {
                        anyhow::bail!("i32.wrap_i64: not enough operands");
                    };

                    value_stack.push(Value::I32(op as i32));
                },

                Instr::I32SConvertF32 => {
                    let Some(Value::F32(op)) = value_stack.pop() else {
                        anyhow::bail!("i32.TKTK: not enough operands");
                    };

                    value_stack.push(Value::I32(op as i32));
                },
                Instr::I32UConvertF32 => {
                    let Some(Value::F32(op)) = value_stack.pop() else {
                        anyhow::bail!("i32.TKTK: not enough operands");
                    };

                    value_stack.push(Value::I32(op as u32 as i32));
                },
                Instr::I32SConvertF64 => {
                    let Some(Value::F64(op)) = value_stack.pop() else {
                        anyhow::bail!("i32.TKTK: not enough operands");
                    };

                    value_stack.push(Value::I32(op as i32));
                },
                Instr::I32UConvertF64 => {
                    let Some(Value::F64(op)) = value_stack.pop() else {
                        anyhow::bail!("i32.TKTK: not enough operands");
                    };

                    value_stack.push(Value::I32(op as u32 as i32));
                },
                Instr::I64SConvertI32 => {
                    let Some(Value::I32(op)) = value_stack.pop() else {
                        anyhow::bail!("i64.extend_i32_s: not enough operands");
                    };

                    value_stack.push(Value::I64(op as i64));
                }
                Instr::I64UConvertI32 => {
                    let Some(Value::I32(op)) = value_stack.pop() else {
                        anyhow::bail!("i64.extend_i32_u: not enough operands");
                    };

                    value_stack.push(Value::I64(op as u32 as i64));
                },
                Instr::I64SConvertF32 => {
                    let Some(Value::F32(op)) = value_stack.pop() else {
                        anyhow::bail!("i64.TKTK: not enough operands");
                    };

                    value_stack.push(Value::I64(op as i64));
                },
                Instr::I64UConvertF32 => {
                    let Some(Value::F32(op)) = value_stack.pop() else {
                        anyhow::bail!("i64.TKTK: not enough operands");
                    };

                    value_stack.push(Value::I64(op as u32 as i64));
                },
                Instr::I64SConvertF64 => {
                    let Some(Value::F64(op)) = value_stack.pop() else {
                        anyhow::bail!("i64.TKTK: not enough operands");
                    };

                    value_stack.push(Value::I64(op as i64));
                },
                Instr::I64UConvertF64 => {
                    let Some(Value::F64(op)) = value_stack.pop() else {
                        anyhow::bail!("i64.TKTK: not enough operands");
                    };

                    value_stack.push(Value::I64(op as u64 as i64));
                },
                Instr::F32SConvertI32 => todo!("F32SConvertI32"),
                Instr::F32UConvertI32 => todo!("F32UConvertI32"),
                Instr::F32SConvertI64 => todo!("F32SConvertI64"),
                Instr::F32UConvertI64 => todo!("F32UConvertI64"),
                Instr::F32ConvertF64 => todo!("F32ConvertF64"),
                Instr::F64SConvertI32 => {
                    let Some(Value::I32(op)) = value_stack.pop() else {
                        anyhow::bail!("f64.convert_i32_u: not enough operands");
                    };

                    value_stack.push(Value::F64(op as f64));
                },
                Instr::F64UConvertI32 => {
                    let Some(Value::I32(op)) = value_stack.pop() else {
                        anyhow::bail!("f64.convert_i32_u: not enough operands");
                    };

                    value_stack.push(Value::F64(op as u32 as f64));
                },
                Instr::F64SConvertI64 => todo!("F64SConvertI64"),
                Instr::F64UConvertI64 => {
                    let Some(Value::I64(op)) = value_stack.pop() else {
                        anyhow::bail!("f64.convert_i64_u: not enough operands");
                    };

                    value_stack.push(Value::F64(op as u64 as f64));
                },
                Instr::F64ConvertF32 => {
                    let Some(Value::F32(op)) = value_stack.pop() else {
                        anyhow::bail!("f64.promote_f32: not enough operands");
                    };

                    value_stack.push(Value::F64(op as f64));
                },
                Instr::I32ReinterpretF32 => todo!("I32ReinterpretF32"),
                Instr::I64ReinterpretF64 => todo!("I64ReinterpretF64"),
                Instr::F32ReinterpretI32 => todo!("F32ReinterpretI32"),
                Instr::F64ReinterpretI64 => todo!("F64ReinterpretI64"),
                Instr::I32SExtendI8 => todo!("I32SExtendI8"),
                Instr::I32SExtendI16 => todo!("I32SExtendI16"),
                Instr::I64SExtendI8 => todo!("I64SExtendI8"),
                Instr::I64SExtendI16 => todo!("I64SExtendI16"),
                Instr::I64SExtendI32 => todo!("I64SExtendI32"),
                Instr::CallHostTrampoline => todo!(),
            }
            frames[frame_idx].pc += 1;
        }

        // TODO: handle multiple return values
        Ok(value_stack)
    }

    pub(crate) fn exports(&self) -> impl Iterator<Item = &str> {
        self.exports.keys().filter_map(|xs| self.internmap.idx(*xs))
    }

    fn compute_constant_expr(&self, module_idx: GuestIndex, instrs: &[Instr]) -> anyhow::Result<Value> {
        Ok(match instrs.first() {
            Some(Instr::F32Const(c)) => Value::F32(*c),
            Some(Instr::F64Const(c)) => Value::F64(*c),
            Some(Instr::I32Const(c)) => Value::I32(*c),
            Some(Instr::I64Const(c)) => Value::I64(*c),
            Some(Instr::GlobalGet(c)) => {
                let globalidx = self.global(module_idx, *c);
                self.global_values[globalidx]
            }

            Some(Instr::RefNull(_c)) => todo!(),
            Some(Instr::RefFunc(c)) => Value::RefFunc(*c),
            _ => anyhow::bail!("unsupported instruction"),
        })
    }
}
