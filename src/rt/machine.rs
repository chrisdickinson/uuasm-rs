use std::{sync::Arc, collections::HashMap, mem::swap};

use crate::{nodes::{Instr, BlockType, Module, MemType, TableType, Type, Data, Code, Export, ExportDesc, ImportDesc, CodeIdx, Elem, Expr, Global, Func}, memory_region::MemoryRegion};

use super::{value::Value, imports::{ExternKey, Extern, GuestIndex, InternMap, ExternFunc, ExternGlobal, ExternTable, ExternMemory}, Imports, function::FuncInst, global::GlobalInst, memory::MemInst, table::TableInst};

pub(super) type InstanceIdx = usize;
pub(super) type MachineCodeIndex = usize;
pub(super) type MachineMemoryIndex = usize;
pub(super) type MachineTableIndex = usize;
pub(super) type MachineGlobalIndex = usize;

struct Frame<'a> {
    #[cfg(test)]
    name: &'static str,
    pc: usize,
    return_unwind_count: usize,
    instrs: &'a [Instr],
    jump_to: Option<usize>,
    block_type: BlockType,
    locals_base_offset: usize,
    instance: InstanceIdx,
}

pub(super) struct Machine<'a> {
    types: Vec<Vec<Type>>,
    code: Vec<Vec<Code>>,
    data: Vec<Vec<Data<'a>>>,
    elements: Vec<Vec<Elem>>,

    functions: Vec<Vec<FuncInst>>,
    globals: Vec<Vec<GlobalInst>>,
    memories: Vec<Vec<MemInst>>,
    tables: Vec<Vec<TableInst>>,

    memory_regions: Vec<MemoryRegion>,
    global_values: Vec<Value>,
    table_instances: Vec<Vec<Value>>,
    exports: HashMap<usize, ExportDesc>,
    internmap: InternMap,
}

impl<'a> Machine<'a> {
    pub(super) fn new(mut imports: Imports<'a>, exports: HashMap<usize, ExportDesc>) -> anyhow::Result<Self> {
        let guests = imports.guests.split_off(0);

        let mut memory_regions: Vec<MemoryRegion> = Vec::new();
        let mut table_instances: Vec<Vec<Value>> = Vec::new();
        let mut global_values: Vec<Value> = Vec::new();

        let mut all_code: Vec<Vec<_>> = Vec::with_capacity(guests.len());
        let mut all_data: Vec<Vec<_>> = Vec::with_capacity(guests.len());
        let mut all_functions: Vec<Vec<_>> = Vec::with_capacity(guests.len());
        let mut all_elements: Vec<Vec<_>> = Vec::with_capacity(guests.len());
        let mut all_globals: Vec<Vec<_>> = Vec::with_capacity(guests.len());
        let mut all_memories: Vec<Vec<_>> = Vec::with_capacity(guests.len());
        let mut all_tables: Vec<Vec<_>> = Vec::with_capacity(guests.len());
        let mut all_types: Vec<Vec<_>> = Vec::with_capacity(guests.len());

        for guest in guests {
            let Module {
                type_section,
                function_section,
                table_section,
                import_section,
                memory_section,
                global_section,
                start_section,
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
            ) =  (
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

            let (
                import_func_count,
                import_mem_count,
                import_table_count,
                import_global_count
            ) = import_section
                .iter()
                .fold((0, 0, 0, 0), |(fc, mc, tc, gc), imp| {
                    match imp.desc {
                        ImportDesc::Func(_) => (fc + 1, mc, tc, gc), 
                        ImportDesc::Mem(_) => (fc, mc + 1, tc, gc),
                        ImportDesc::Table(_) => (fc, mc, tc + 1, gc),
                        ImportDesc::Global(_) => (fc, mc, tc, gc + 1),
                    }
                });

            let mut functions = Vec::with_capacity(function_section.len() + import_func_count);
            let mut globals = Vec::with_capacity(global_section.len() + import_global_count);
            let mut memories = Vec::with_capacity(memory_section.len() + import_mem_count);
            let mut tables = Vec::with_capacity(table_section.len() + import_table_count);
            for imp in import_section {
                match imp.desc {
                    ImportDesc::Func(desc) => { functions.push(FuncInst::resolve(desc, &imp, &imports)?); }
                    ImportDesc::Mem(desc) => { memories.push(MemInst::resolve(desc, &imp, &imports)?); }
                    ImportDesc::Table(desc) => { tables.push(TableInst::resolve(desc, &imp, &imports)?); }
                    ImportDesc::Global(desc) => { globals.push(GlobalInst::resolve(desc, &imp, &imports)?); }
                }
            }

            let mut saw_error = false;
            let code_max = code_section.len();
            functions.extend(function_section.into_iter().enumerate().map(|(code_idx, xs)| {
                if code_idx >= code_max {
                    saw_error = true;
                }
                FuncInst::new(xs, CodeIdx(code_idx as u32))
            }));

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
                GlobalInst::new(global_type, global_base_offset + idx, instrs.into_boxed_slice())
            }));

            all_code.push(code_section);
            all_data.push(data_section);
            all_elements.push(element_section);
            all_functions.push(functions);
            all_globals.push(globals);
            all_memories.push(memories);
            all_tables.push(tables);
            all_types.push(type_section);
        }

        Ok(Self {
            types: all_types,
            code: all_code,
            data: all_data,
            elements: all_elements,
            functions: all_functions,
            globals: all_globals,
            memories: all_memories,
            tables: all_tables,
            memory_regions,
            global_values,
            table_instances,
            exports,
            internmap: imports.internmap
        })
    }

    fn resolve(&self, extern_value: &Extern) -> Option<usize> {
        match extern_value {
            Extern::Func(ExternFunc::Host(v)) |
            Extern::Global(ExternGlobal::Host(v)) |
            Extern::Table(ExternTable::Host(v)) |
            Extern::Memory(ExternMemory::Host(v)) => Some(*v),

            _ => None
        }
    }

    pub(super) fn code(&self, module_idx: usize, code_idx: usize) -> Option<&Code> {
        self.code.get(module_idx)?.get(code_idx)
    }

    pub(super) fn function(&self, module_idx: usize, func_idx: usize) -> Option<&FuncInst> {
        self.functions.get(module_idx)?.get(func_idx)
    }

    pub(crate) fn call(&mut self, funcname: &str, args: &[Value]) -> anyhow::Result<Vec<Value>> {
        // how to get lookup funcname? (have to preserve the last module's externs)
        let func_idx = self.internmap.get(funcname)
            .ok_or_else(|| anyhow::anyhow!("no such export {funcname}"))?;

        let function = self.functions.last()
            .ok_or_else(|| anyhow::anyhow!("missing final module"))?
            .get(func_idx)
            .ok_or_else(|| anyhow::anyhow!(""))?
            ;

        let func = function.func(&self, self.functions.len() - 1);

        todo!()
    }

    pub(crate) fn exports(&self) -> impl Iterator<Item = &str> {
        self.exports.keys().filter_map(|xs| self.internmap.idx(*xs))
    }
}
