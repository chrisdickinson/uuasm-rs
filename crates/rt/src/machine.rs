use std::{
    collections::{HashMap, HashSet},
    mem,
    num::NonZeroU64,
    ops::{Deref, DerefMut, Index, IndexMut, Range},
    sync::{Arc, Mutex},
};

use anyhow::Context;
use smallvec::{smallvec, SmallVec};

use crate::{
    locals::StackMapStack,
    memory_region::MemoryRegion,
    stack::{PageSizedStack, StackValue},
    value::RefValue,
};
use crate::{
    prelude::*,
    stack::{Fence, Stack},
};
use uuasm_ir::{
    BlockType, ByteVec, Code, CodeIdx, Data, Elem, ElemIdx, ElemMode, Export, ExportDesc, Expr,
    Func, FuncIdx, Global, GlobalIdx, Import, ImportDesc, Instr, LabelIdx, LocalIdx, MemIdx,
    Module, ModuleBuilder, ModuleIntoInner, Name, RefType, ResultType, TableIdx, TableType, Type,
    TypeIdx, ValType,
};

use super::{
    function::FuncInst,
    global::{GlobalInst, GlobalInstImpl},
    imports::{Extern, ExternKey, GuestIndex, LookupImport},
    memory::{MemInst, MemoryInstImpl},
    table::{TableInst, TableInstImpl},
    value::Value,
};
use crate::intern_map::InternMap;

pub(super) type InstanceIdx = usize;
pub(super) type MachineCodeIndex = usize;
pub(super) type MachineMemoryIndex = usize;
pub(super) type MachineTableIndex = usize;
pub(super) type MachineGlobalIndex = usize;

trait ElemValue {
    fn value(
        &self,
        idx: usize,
        module_idx: usize,
        machine: &Machine,
        resources: &mut Resources,
    ) -> anyhow::Result<StackValue>;
}

impl ElemValue for Elem {
    fn value(
        &self,
        idx: usize,
        module_idx: usize,
        machine: &Machine,
        resources: &mut Resources,
    ) -> anyhow::Result<StackValue> {
        machine.compute_constant_expr(module_idx, self.exprs[idx].0.as_slice(), resources)
    }
}

#[derive(Debug)]
struct Frame<'a> {
    #[cfg(test)]
    name: &'static str,
    pc: usize,
    fence: Fence,

    // how far are we from the call frame?
    return_unwind_count: usize,

    // what instructions are we operating on?
    instrs: &'a [Instr],

    jump_to_start_of_instrs: bool,

    params: SmallVec<[ValType; 2]>,
    results: SmallVec<[ValType; 2]>,

    guest_index: GuestIndex,
}

pub type FuncWrap = Arc<dyn Fn(&[Value], &mut [Value]) -> anyhow::Result<()> + Send + Sync>;

#[derive(Clone)]
pub(crate) struct ExternalFunction {
    pub(crate) func: FuncWrap,
    pub(crate) typedef: Type,
}

impl std::fmt::Debug for ExternalFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalFunction")
            .field("typedef", &self.typedef)
            .finish()
    }
}

pub struct Table {
    kind: TableType,
    values: SmallVec<[RefValue; 4]>,
}

impl IndexMut<u32> for Table {
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        &mut self.values[index as usize]
    }
}

impl Index<Range<usize>> for Table {
    type Output = [RefValue];

    fn index(&self, index: Range<usize>) -> &Self::Output {
        self.values.index(index)
    }
}

impl IndexMut<Range<usize>> for Table {
    fn index_mut(&mut self, index: Range<usize>) -> &mut [RefValue] {
        self.values.index_mut(index)
    }
}

impl Index<u32> for Table {
    type Output = RefValue;

    fn index(&self, index: u32) -> &Self::Output {
        unsafe { self.values.get_unchecked(index as usize) }
    }
}

impl Table {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    fn grow(&mut self, delta: i32, fill: RefValue) -> i32 {
        let len = self.values.len();

        let new_size = len.saturating_add(delta as usize);
        let max = self.kind.1.max().unwrap_or(u32::MAX);
        if new_size > max as usize {
            return -1;
        }
        self.values.resize(new_size, fill);
        len as i32
    }
}

// Really, this per-"execution" data is a store
pub(crate) struct Resources {
    // TODO: turn this into a Stack
    global_values: Vec<StackValue>,
    memory_regions: Vec<MemoryRegion>,
    table_instances: Vec<Table>,

    dropped_elements: HashSet<(usize, usize)>,
    dropped_data: HashSet<(usize, usize)>,
    external_functions: Vec<ExternalFunction>,
}

// and this should be the thing that holds platonic information about
// modules
pub struct Machine {
    initialized: HashSet<usize>,

    imports: Vec<Box<[Import]>>,
    exports: Vec<Box<[Export]>>,
    types: Vec<Box<[Type]>>,
    code: Vec<Box<[Code]>>,
    data: Vec<Box<[Data]>>,
    elements: Vec<Box<[Elem]>>,

    functions: Vec<Box<[FuncInst]>>,
    globals: Vec<Box<[GlobalInst]>>,
    memories: Vec<Box<[MemInst]>>,
    tables: Vec<Box<[TableInst]>>,
    starts: Vec<Option<FuncIdx>>,

    resources: Arc<Mutex<Resources>>,

    modname_to_guest_idx: HashMap<usize, HashSet<usize>>,
    externs: HashMap<ExternKey, Extern>,
    internmap: InternMap,
}

impl LookupImport for Machine {
    fn lookup(&self, import: &Import) -> Option<Extern> {
        let modname = self.internmap.get(import.module())?;
        let name = self.internmap.get(import.name())?;
        let key = ExternKey(modname, name);
        self.externs.get(&key).copied()
    }
}

impl Machine {
    fn link_extern(&mut self, modname: &str, name: &str, ext: Extern) {
        let modname = self.internmap.insert(modname);
        let name = self.internmap.insert(name);

        self.externs.insert(ExternKey(modname, name), ext);
    }

    pub fn link_module(&mut self, modname: &str, module: Module) -> anyhow::Result<()> {
        let idx = self.types.len(); // any of these will do.

        let modname_idx = self.internmap.insert(modname);
        self.modname_to_guest_idx
            .entry(modname_idx)
            .or_default()
            .insert(idx);

        let arc_resources = self.resources.clone();
        let mut resource_lock = arc_resources
            .try_lock()
            .map_err(|_| anyhow::anyhow!("failed to lock resources"))?;
        let resources = resource_lock.deref_mut();

        for export in module.export_section().unwrap_or_default() {
            match export.desc() {
                ExportDesc::Func(ref func_idx) => {
                    self.link_extern(modname, export.name(), Extern::Func(idx, *func_idx));
                }
                ExportDesc::Table(table_idx) => {
                    self.link_extern(modname, export.name(), Extern::Table(idx, *table_idx));
                }
                ExportDesc::Mem(mem_idx) => {
                    self.link_extern(modname, export.name(), Extern::Memory(idx, *mem_idx));
                }
                ExportDesc::Global(global_idx) => {
                    self.link_extern(modname, export.name(), Extern::Global(idx, *global_idx));
                }
            }
        }

        self.link_guest(module, resources)
    }

    fn link_guest(&mut self, guest: Module, resources: &mut Resources) -> anyhow::Result<()> {
        let ModuleIntoInner {
            type_section,
            function_section,
            table_section,
            import_section,
            memory_section,
            global_section,
            element_section,
            code_section,
            data_section,
            export_section,
            start_section,
            ..
        } = guest.into_inner();

        let (
            type_section,
            function_section,
            import_section,
            export_section,
            table_section,
            memory_section,
            global_section,
            element_section,
            code_section,
            data_section,
            start_section,
        ) = (
            type_section.map(|xs| xs.inner).unwrap_or_default(),
            function_section.map(|xs| xs.inner).unwrap_or_default(),
            import_section.map(|xs| xs.inner).unwrap_or_default(),
            export_section.map(|xs| xs.inner).unwrap_or_default(),
            table_section.map(|xs| xs.inner).unwrap_or_default(),
            memory_section.map(|xs| xs.inner).unwrap_or_default(),
            global_section.map(|xs| xs.inner).unwrap_or_default(),
            element_section.map(|xs| xs.inner).unwrap_or_default(),
            code_section.map(|xs| xs.inner).unwrap_or_default(),
            data_section.map(|xs| xs.inner).unwrap_or_default(),
            start_section.map(|xs| xs.inner),
        );

        let (import_func_count, import_mem_count, import_table_count, import_global_count) =
            import_section
                .iter()
                .fold((0, 0, 0, 0), |(fc, mc, tc, gc), imp| match imp.desc() {
                    ImportDesc::Func(_) => (fc + 1, mc, tc, gc),
                    ImportDesc::Mem(_) => (fc, mc + 1, tc, gc),
                    ImportDesc::Table(_) => (fc, mc, tc + 1, gc),
                    ImportDesc::Global(_) => (fc, mc, tc, gc + 1),
                });

        let mut functions = Vec::with_capacity(function_section.len() + import_func_count);
        let mut globals = Vec::with_capacity(global_section.len() + import_global_count);
        let mut memories = Vec::with_capacity(memory_section.len() + import_mem_count);
        let mut tables = Vec::with_capacity(table_section.len() + import_table_count);
        for imp in import_section.iter() {
            match imp.desc() {
                ImportDesc::Func(desc) => {
                    functions.push(FuncInst::resolve(*desc, imp, self)?);
                }

                ImportDesc::Mem(desc) => {
                    memories.push(MemInst::resolve(*desc, imp, self)?);
                }

                ImportDesc::Table(desc) => {
                    tables.push(TableInst::resolve(*desc, imp, self)?);
                }

                ImportDesc::Global(desc) => {
                    globals.push(GlobalInst::resolve(*desc, imp, self)?);
                }
            }
        }

        let mut saw_error = false;
        let code_max = code_section.len();
        functions.extend(IntoIterator::into_iter(function_section).enumerate().map(
            |(code_idx, xs)| {
                if code_idx >= code_max {
                    saw_error = true;
                }
                FuncInst::new(xs, CodeIdx(code_idx as u32))
            },
        ));

        if saw_error {
            anyhow::bail!("code idx out of range");
        }

        memories.extend(IntoIterator::into_iter(memory_section).map(|memtype| {
            let memory_region_idx = resources.memory_regions.len();
            resources.memory_regions.push(MemoryRegion::new(memtype.0));
            MemInst::new(memtype, memory_region_idx)
        }));

        tables.extend(IntoIterator::into_iter(table_section).map(|tabletype| {
            let table_instance_idx = resources.table_instances.len();
            resources.table_instances.push(Table {
                values: (0..tabletype.1.min()).map(|_| None).collect(),
                kind: tabletype,
            });
            TableInst::new(tabletype, table_instance_idx)
        }));

        let global_base_offset = resources.global_values.len();
        resources.global_values.reserve(global_section.len());

        globals.extend(
            IntoIterator::into_iter(global_section)
                .enumerate()
                .map(|(idx, global)| {
                    let Global(global_type, Expr(instrs)) = global;
                    resources
                        .global_values
                        .push(global_type.0.instantiate_stackvalue());
                    GlobalInst::new(
                        global_type,
                        global_base_offset + idx,
                        instrs.into_boxed_slice(),
                    )
                }),
        );

        self.starts.push(start_section);
        self.imports.push(import_section);
        self.exports.push(export_section);
        self.code.push(code_section);
        self.data.push(data_section);
        self.elements.push(element_section);
        self.functions.push(functions.into_boxed_slice());
        self.globals.push(globals.into_boxed_slice());
        self.memories.push(memories.into_boxed_slice());
        self.tables.push(tables.into_boxed_slice());
        self.types.push(type_section);
        Ok(())
    }

    pub(super) fn new(
        guests: Vec<Module>,
        exports: HashMap<ExternKey, Extern>,
        intern_map: InternMap,
        external_functions: Vec<ExternalFunction>,
        modname_to_guest_idx: HashMap<usize, HashSet<usize>>,
    ) -> anyhow::Result<Self> {
        let resources = Arc::new(Mutex::new(Resources {
            memory_regions: Vec::with_capacity(4),
            global_values: Vec::new(),
            table_instances: Vec::with_capacity(4),
            dropped_data: HashSet::new(),
            dropped_elements: HashSet::new(),
            external_functions,
        }));
        let mut machine = Self {
            initialized: HashSet::new(),
            exports: Vec::with_capacity(guests.len()),
            imports: Vec::with_capacity(guests.len()),
            code: Vec::with_capacity(guests.len()),
            data: Vec::with_capacity(guests.len()),
            functions: Vec::with_capacity(guests.len()),
            elements: Vec::with_capacity(guests.len()),
            globals: Vec::with_capacity(guests.len()),
            memories: Vec::with_capacity(guests.len()),
            tables: Vec::with_capacity(guests.len()),
            types: Vec::with_capacity(guests.len()),
            starts: Vec::with_capacity(guests.len()),
            resources,
            externs: exports,
            modname_to_guest_idx,
            internmap: intern_map,
        };

        let arc_resources = machine.resources.clone();
        let mut resource_lock = arc_resources
            .try_lock()
            .map_err(|_| anyhow::anyhow!("failed to lock resources"))?;
        let resources = resource_lock.deref_mut();

        for guest in guests {
            machine.link_guest(guest, resources)?;
        }

        // TODO: flatten references!
        Ok(machine)
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

    fn code(&self, module_idx: GuestIndex, CodeIdx(code_idx): CodeIdx) -> &Code {
        &self.code[module_idx][code_idx as usize]
    }

    fn function(
        &self,
        mut module_idx: GuestIndex,
        mut func_idx: FuncIdx,
    ) -> Option<(GuestIndex, &FuncInst)> {
        loop {
            let func = self.functions.get(module_idx)?.get(func_idx.0 as usize)?;

            match func.r#impl {
                super::function::FuncInstImpl::Local(_) => return Some((module_idx, func)),
                super::function::FuncInstImpl::Remote(midx, fidx) => {
                    module_idx = midx;
                    func_idx = fidx;
                }
            }
        }
    }

    #[inline]
    fn typedef(&self, guest_idx: GuestIndex, type_idx: TypeIdx) -> Option<&Type> {
        self.types.get(guest_idx)?.get(type_idx.0 as usize)
    }

    pub(crate) fn initialize(&mut self, at_idx: usize) -> anyhow::Result<()> {
        if self.initialized.contains(&at_idx) {
            return Ok(());
        }
        self.initialized.insert(at_idx);

        let mut initializers = Vec::new();
        for import in self.imports[at_idx].iter() {
            let Some(idx) = self.internmap.get(import.module()) else {
                continue;
            };
            let Some(guest_indices) = self.modname_to_guest_idx.get(&idx) else {
                continue;
            };
            initializers.extend(guest_indices.iter().copied());
        }
        for guest_idx in initializers {
            self.initialize(guest_idx)?;
        }

        let mut resource_lock = self
            .resources
            .try_lock()
            .map_err(|_| anyhow::anyhow!("failed to lock resources"))?;
        let resources = resource_lock.deref_mut();
        // loop over the imports of the current idx. call initialize on them, then init ourselves.

        for global in self.globals[at_idx].iter() {
            let Some((global_idx, instrs)) = global.initdata() else {
                continue;
            };

            resources.global_values[global_idx] =
                self.compute_constant_expr(at_idx, instrs, &mut *resources)?;
        }

        for (data_idx, data) in self.data[at_idx].iter().enumerate() {
            match data {
                Data::Active(ByteVec(data), memory_idx, expr) => {
                    let memoffset = self.compute_constant_expr(at_idx, &expr.0, &mut *resources)?;
                    let memoffset = memoffset
                        .as_i32()
                        .ok_or_else(|| anyhow::anyhow!("expected i32"))?
                        as u32 as usize;

                    let memoryidx = self.memory(at_idx, *memory_idx);
                    let memory = resources
                        .memory_regions
                        .get_mut(memoryidx)
                        .ok_or_else(|| anyhow::anyhow!("no such memory"))?;

                    if memory.grow_to_fit(data, memoffset).is_err() {
                        anyhow::bail!("out of bounds memory access")
                    }

                    if memoffset.saturating_add(data.len()) > memory.len() {
                        anyhow::bail!("out of bounds memory access")
                    }

                    memory.copy_data(data, memoffset);
                    resources.dropped_data.insert((at_idx, data_idx));
                }
                Data::Passive(_) => continue,
            }
        }

        let mut active_elems = Vec::new();
        for (elem_idx, elem) in self.elements[at_idx].iter_mut().enumerate() {
            if !matches!(&elem.mode, ElemMode::Active { .. } | ElemMode::Declarative) {
                continue;
            }

            let mut empty = Elem {
                mode: ElemMode::Passive,
                kind: elem.kind,
                exprs: Box::new([]),
                flags: elem.flags,
            };

            resources.dropped_elements.insert((at_idx, elem_idx));
            mem::swap(elem, &mut empty);

            if !matches!(&empty.mode, ElemMode::Active { .. }) {
                continue;
            }
            active_elems.push(empty);
        }

        for elem in active_elems {
            let ElemMode::Active { table_idx, offset } = &elem.mode else {
                continue;
            };
            let offset = self.compute_constant_expr(at_idx, &offset.0, &mut *resources)?;
            let offset = offset
                .as_i32()
                .ok_or_else(|| anyhow::anyhow!("expected i32"))? as u32
                as usize;

            let values = elem
                .exprs
                .iter()
                .map(|xs| self.compute_constant_expr(at_idx, &xs.0, &mut *resources))
                .collect::<anyhow::Result<Vec<_>>>()?;

            let tableidx = self.table(at_idx, *table_idx);
            let table = resources
                .table_instances
                .get_mut(tableidx)
                .ok_or_else(|| anyhow::anyhow!("no such table"))?;

            if values.len() + offset > table.len() {
                anyhow::bail!("out of bounds table access");
            }

            for (idx, xs) in values.iter().enumerate() {
                table[(idx + offset) as u32] = xs.as_ref_value();
            }
        }

        drop(resource_lock);

        // TODO: do we have a start section? an initialize export? call them.
        if let Some(start_idx) = self.starts[at_idx] {
            self.call_funcidx(&[at_idx], (at_idx, start_idx), &[])?;
        }

        Ok(())
    }

    pub(crate) fn alias(&mut self, modname: &str, to_modname: &str) -> anyhow::Result<()> {
        // get all of the exports of "modname" and create
        let modname_idx = self
            .internmap
            .get(modname)
            .ok_or_else(|| anyhow::anyhow!("no such instance {modname}"))?;

        let module_indices = self
            .modname_to_guest_idx
            .get(&modname_idx)
            .into_iter()
            .map(|xs| xs.iter())
            .flat_map(|xs| xs.copied());

        let mut seen_names = HashSet::new();

        let mut types = Vec::new();
        let mut imports = Vec::new();
        let mut exports = Vec::new();

        // The relative object indices in our new module will correspond to these counts; since
        // we may be aliasing across multiple modules we need to keep track of these ourselves (we
        // can't just clone the export desc.)
        let mut func_count = 0;
        let mut table_count = 0;
        let mut mem_count = 0;
        let mut global_count = 0;
        for idx in module_indices {
            for export in self.exports[idx].iter() {
                if !seen_names.insert(self.internmap.get(export.name()).unwrap()) {
                    anyhow::bail!("cannot alias; {} is exported multiple times", export.name());
                }
                let (export_desc, import_desc) = match export.desc() {
                    ExportDesc::Func(func_idx) => {
                        let func = &self.functions[idx][func_idx.0 as usize];
                        let type_idx = func.typeidx();

                        let type_idx_out = types.len();
                        types.push(self.types[idx][type_idx.0 as usize].clone());
                        func_count += 1;
                        (
                            ExportDesc::Func(FuncIdx(func_count - 1)),
                            ImportDesc::Func(TypeIdx(type_idx_out as u32)),
                        )
                    }
                    ExportDesc::Table(table_idx) => {
                        let table = &self.tables[idx][table_idx.0 as usize];

                        table_count += 1;
                        (
                            ExportDesc::Table(TableIdx(table_count - 1)),
                            ImportDesc::Table(*table.typedef()),
                        )
                    }
                    ExportDesc::Mem(mem_idx) => {
                        let memory = &self.memories[idx][mem_idx.0 as usize];
                        mem_count += 1;
                        (
                            ExportDesc::Mem(MemIdx(mem_count - 1)),
                            ImportDesc::Mem(*memory.typedef()),
                        )
                    }
                    ExportDesc::Global(global_idx) => {
                        let global = &self.globals[idx][global_idx.0 as usize];
                        global_count += 1;
                        (
                            ExportDesc::Global(GlobalIdx(global_count - 1)),
                            ImportDesc::Global(*global.typedef()),
                        )
                    }
                };

                imports.push(Import::new(
                    Name(modname.to_string()),
                    Name(export.name().to_string()),
                    import_desc,
                ));
                exports.push(Export::new(export.name().to_string(), export_desc));
            }
        }

        // TODO: do we clone the "start" section?
        let module = ModuleBuilder::new()
            .type_section(types.into())
            .import_section(imports.into_boxed_slice())
            .export_section(exports.into_boxed_slice())
            .build();

        self.link_module(to_modname, module)
    }

    pub(crate) fn init(&mut self, modname: &str) -> anyhow::Result<()> {
        let modname_idx = self
            .internmap
            .get(modname)
            .ok_or_else(|| anyhow::anyhow!("no such instance {modname}"))?;

        let mod_indices = self
            .modname_to_guest_idx
            .get(&modname_idx)
            .map(|xs| xs.iter().copied().collect::<Vec<_>>())
            .unwrap_or_default();

        for module_idx in mod_indices {
            self.initialize(module_idx)?;
        }

        Ok(())
    }

    pub fn call(
        &mut self,
        modname: &str,
        funcname: &str,
        args: &[Value],
    ) -> anyhow::Result<Vec<Value>> {
        let funcname_idx = self
            .internmap
            .get(funcname)
            .ok_or_else(|| anyhow::anyhow!("no such export {funcname}"))?;

        let modname_idx = self
            .internmap
            .get(modname)
            .ok_or_else(|| anyhow::anyhow!("no such instance {modname}"))?;

        let mod_indices = self
            .modname_to_guest_idx
            .get(&modname_idx)
            .map(|xs| xs.iter().copied().collect::<Vec<_>>())
            .unwrap_or_default();

        let Extern::Func(module_idx, func_idx) = self
            .externs
            .get(&ExternKey(modname_idx, funcname_idx))
            .ok_or_else(|| anyhow::anyhow!("no such export {funcname}"))?
        else {
            anyhow::bail!("export {funcname} is not a function");
        };

        #[cfg(any())]
        eprintln!("call {modname} {funcname} = = = = = = = = = = = = = = = = = = = = =");
        self.call_funcidx(mod_indices.as_slice(), (*module_idx, *func_idx), args)
    }

    fn call_funcidx(
        &mut self,
        module_indices: &[usize],
        target: (usize, FuncIdx),
        args: &[Value],
    ) -> anyhow::Result<Vec<Value>> {
        let (module_idx, func_idx) = target;
        for module_idx in module_indices {
            self.initialize(*module_idx)?;
        }

        let mut resource_lock = self
            .resources
            .try_lock()
            .map_err(|_| anyhow::anyhow!("failed to lock resources"))?;
        let mut resources = resource_lock.deref_mut();

        let (module_idx, function) = self
            .function(module_idx, func_idx)
            .ok_or_else(|| anyhow::anyhow!("missing final module"))?;

        let Type(ResultType(params), ResultType(results)) = self
            .typedef(module_idx, *function.typeidx())
            .ok_or_else(|| anyhow::anyhow!("missing typedef"))?;

        if args.len() < params.len() {
            anyhow::bail!("not enough arguments; expected {} args", params.len());
        }

        if args.len() > params.len() {
            anyhow::bail!("too many arguments; expected {} args", params.len());
        }

        let mut value_stack = PageSizedStack::new();
        let mut locals = StackMapStack::new();

        for (idx, (param_type, value)) in params.iter().zip(args.iter()).enumerate() {
            param_type
                .validate(value)
                .with_context(|| format!("bad argument at {}", idx))?;

            let value: StackValue = (*value).into();
            value_stack.push_value(value);
        }

        let Code(Func {
            locals: local_defs,
            expr: Expr(instrs),
        }) = self.code(module_idx, function.codeidx());

        locals.begin_call(&mut value_stack, params, local_defs);

        let mut frames = Vec::<Frame<'_>>::new();
        let mut frame = Frame {
            #[cfg(test)]
            name: "init",
            pc: 0,
            return_unwind_count: 0,
            instrs,
            guest_index: module_idx,
            fence: value_stack.fence(),
            params: params.iter().copied().collect(),
            results: results.iter().copied().collect(),
            jump_to_start_of_instrs: false,
        };

        #[cfg(any())]
        eprintln!("call = = = = = = = = = = = = = = = = = = = = =");
        loop {
            if frame.pc >= frame.instrs.len() {
                // we're exiting a block normally. due to validation, the only values left on the
                // stack are the ones that are supposed to be there. HOWEVER, we may be returning
                // to a callsite, in which case we need to .unwind() the local stack as well. (So
                // do that!!)
                //
                // TKTK: maybe we need separate arrays for "program counter", "return frame
                // length", etc? or we could store return count etc in the stack after the locals
                //
                // scratch space here:
                //
                // - locals must be unwound after calls
                // - globals must be unwound after cross module calls (assuming that's how we
                //   implement globals)
                // - otherwise we can leave the value stack as-is
                // - however if we return we can handle the unwinding there (should we?)
                // - if we br to a block we'll pop values corresponding to the types, then
                //   unwind to the block fence
                //      - it'd be interesting to annotate blocks with "is br target" in validation
                //        but maybe we wait until we have more info via benchmarks
                //
                // So. Which problem are we solving first? Are we solving br, return, or "ran out
                // of instrs" first? Or are we solving globals? Which problem is most fundamental?
                //
                // Globals seem more fundamental. Therefore.

                if frame.return_unwind_count == 0 {
                    locals.end_call(&mut value_stack, &frame.results);
                } else {
                    // idk if we even have to unwind??
                    // value_stack.unwind(frame.fence);
                }

                let Some(new_frame) = frames.pop() else { break };

                frame = new_frame;
                continue;
            }

            match &frame.instrs[frame.pc] {
                Instr::Unreachable => anyhow::bail!("unreachable"),
                Instr::Nop => {}

                Instr::Block(block_type, blockinstrs) => {
                    let new_frame = Frame {
                        #[cfg(test)]
                        name: "block",
                        pc: 0,
                        fence: value_stack.fence(),
                        return_unwind_count: frame.return_unwind_count + 1,
                        instrs: blockinstrs,
                        jump_to_start_of_instrs: false,
                        guest_index: frame.guest_index,
                        params: match block_type {
                            BlockType::Empty => SmallVec::new(),
                            BlockType::Val(val_type) => smallvec![*val_type],
                            BlockType::TypeIndex(TypeIdx(type_idx)) => self.types
                                [frame.guest_index][*type_idx as usize]
                                .0
                                 .0
                                .iter()
                                .copied()
                                .collect(),
                        },
                        results: match block_type {
                            BlockType::Empty => SmallVec::new(),
                            BlockType::Val(val_type) => smallvec![*val_type],
                            BlockType::TypeIndex(TypeIdx(type_idx)) => self.types
                                [frame.guest_index][*type_idx as usize]
                                .1
                                 .0
                                .iter()
                                .copied()
                                .collect(),
                        },
                    };
                    frame.pc += 1;
                    let old_frame = frame;
                    frames.push(old_frame);
                    frame = new_frame;
                    continue;
                }

                Instr::Loop(block_type, blockinstrs) => {
                    let new_frame = Frame {
                        #[cfg(test)]
                        name: "Loop",
                        pc: 0,
                        return_unwind_count: frame.return_unwind_count + 1,
                        instrs: blockinstrs,
                        jump_to_start_of_instrs: true,
                        fence: value_stack.fence(),
                        guest_index: frame.guest_index,
                        params: match block_type {
                            BlockType::Empty => SmallVec::new(),
                            BlockType::Val(val_type) => smallvec![*val_type],
                            BlockType::TypeIndex(TypeIdx(type_idx)) => self.types
                                [frame.guest_index][*type_idx as usize]
                                .0
                                 .0
                                .iter()
                                .copied()
                                .collect(),
                        },
                        results: match block_type {
                            BlockType::Empty => SmallVec::new(),
                            BlockType::Val(val_type) => smallvec![*val_type],
                            BlockType::TypeIndex(TypeIdx(type_idx)) => self.types
                                [frame.guest_index][*type_idx as usize]
                                .1
                                 .0
                                .iter()
                                .copied()
                                .collect(),
                        },
                    };
                    frame.pc += 1;
                    let old_frame = frame;
                    frames.push(old_frame);
                    frame = new_frame;
                    continue;
                }

                Instr::If(block_type, consequent) => {
                    let test = value_stack.pop::<i32>();
                    let blockinstrs = if test == 0 { &[] } else { consequent.deref() };

                    let new_frame = Frame {
                        #[cfg(test)]
                        name: "If",
                        pc: 0,
                        return_unwind_count: frame.return_unwind_count + 1,
                        instrs: blockinstrs,
                        jump_to_start_of_instrs: false,
                        fence: value_stack.fence(),
                        guest_index: frame.guest_index,
                        params: match block_type {
                            BlockType::Empty => SmallVec::new(),
                            BlockType::Val(val_type) => smallvec![*val_type],
                            BlockType::TypeIndex(TypeIdx(type_idx)) => self.types
                                [frame.guest_index][*type_idx as usize]
                                .0
                                 .0
                                .iter()
                                .copied()
                                .collect(),
                        },
                        results: match block_type {
                            BlockType::Empty => SmallVec::new(),
                            BlockType::Val(val_type) => smallvec![*val_type],
                            BlockType::TypeIndex(TypeIdx(type_idx)) => self.types
                                [frame.guest_index][*type_idx as usize]
                                .1
                                 .0
                                .iter()
                                .copied()
                                .collect(),
                        },
                    };
                    frame.pc += 1;
                    let old_frame = frame;
                    frames.push(old_frame);
                    frame = new_frame;
                    continue;
                }

                Instr::IfElse(block_type, consequent, alternate) => {
                    let test = value_stack.pop::<i32>();
                    let blockinstrs = if test == 0 {
                        alternate.deref()
                    } else {
                        consequent.deref()
                    };

                    let new_frame = Frame {
                        #[cfg(test)]
                        name: "IfElse",
                        pc: 0,
                        return_unwind_count: frame.return_unwind_count + 1,
                        instrs: blockinstrs,
                        jump_to_start_of_instrs: false,
                        fence: value_stack.fence(),
                        guest_index: frame.guest_index,
                        params: match block_type {
                            BlockType::Empty => SmallVec::new(),
                            BlockType::Val(val_type) => smallvec![*val_type],
                            BlockType::TypeIndex(TypeIdx(type_idx)) => self.types
                                [frame.guest_index][*type_idx as usize]
                                .0
                                 .0
                                .iter()
                                .copied()
                                .collect(),
                        },
                        results: match block_type {
                            BlockType::Empty => SmallVec::new(),
                            BlockType::Val(val_type) => smallvec![*val_type],
                            BlockType::TypeIndex(TypeIdx(type_idx)) => self.types
                                [frame.guest_index][*type_idx as usize]
                                .1
                                 .0
                                .iter()
                                .copied()
                                .collect(),
                        },
                    };
                    frame.pc += 1;
                    let old_frame = frame;
                    frames.push(old_frame);
                    frame = new_frame;
                    continue;
                }

                Instr::Br(LabelIdx(idx)) => {
                    // look at the arity of the target block type. preserve that many values from
                    // the stack.
                    if *idx > 0 {
                        frames.truncate(frames.len() - (idx - 1) as usize);
                        frame = frames
                            .pop()
                            .ok_or_else(|| anyhow::anyhow!("stack underflow"))?;
                    }

                    frame.pc = if frame.jump_to_start_of_instrs {
                        0
                    } else {
                        frame.instrs.len()
                    };

                    let types = if frame.jump_to_start_of_instrs {
                        &frame.params
                    } else {
                        &frame.results
                    };

                    let values: SmallVec<[StackValue; 2]> = types
                        .iter()
                        .rev()
                        .map(|vt| value_stack.pop_valtype(*vt))
                        .collect();
                    value_stack.unwind(frame.fence);
                    for value in values.into_iter().rev() {
                        value_stack.push_value(value);
                    }

                    continue;
                }

                Instr::BrIf(LabelIdx(idx)) => {
                    let test = value_stack.pop::<i32>();
                    if test != 0 {
                        if *idx > 0 {
                            frames.truncate(frames.len() - (idx - 1) as usize);
                            frame = frames
                                .pop()
                                .ok_or_else(|| anyhow::anyhow!("stack underflow"))?;
                        }

                        frame.pc = if frame.jump_to_start_of_instrs {
                            0
                        } else {
                            frame.instrs.len()
                        };

                        let types = if frame.jump_to_start_of_instrs {
                            &frame.params
                        } else {
                            &frame.results
                        };

                        let values: SmallVec<[StackValue; 2]> = types
                            .iter()
                            .rev()
                            .map(|vt| value_stack.pop_valtype(*vt))
                            .collect();
                        value_stack.unwind(frame.fence);
                        for value in values.into_iter().rev() {
                            value_stack.push_value(value);
                        }

                        continue;
                    }
                }

                Instr::BrTable(labels, LabelIdx(alternate)) => {
                    let test = value_stack.pop::<i32>();

                    let v = test as usize;
                    let idx = if v >= labels.len() {
                        *alternate
                    } else {
                        let LabelIdx(idx) = labels[v];
                        idx
                    };

                    if idx > 0 {
                        frames.truncate(frames.len() - (idx - 1) as usize);
                        frame = frames
                            .pop()
                            .ok_or_else(|| anyhow::anyhow!("stack underflow"))?;
                    }

                    frame.pc = if frame.jump_to_start_of_instrs {
                        0
                    } else {
                        frame.instrs.len()
                    };

                    let types = if frame.jump_to_start_of_instrs {
                        &frame.params
                    } else {
                        &frame.results
                    };

                    let values: SmallVec<[StackValue; 2]> = types
                        .iter()
                        .rev()
                        .map(|vt| value_stack.pop_valtype(*vt))
                        .collect();
                    value_stack.unwind(frame.fence);
                    for value in values.into_iter().rev() {
                        value_stack.push_value(value);
                    }

                    continue;
                }

                Instr::Return => {
                    let ruc = frame.return_unwind_count;
                    if ruc > 0 {
                        frames.truncate(frames.len() - (ruc - 1));
                        frame = frames
                            .pop()
                            .ok_or_else(|| anyhow::anyhow!("stack underflow"))?;
                    }

                    frame.pc = frame.instrs.len();
                    continue;
                }

                Instr::Call(func_idx) => {
                    let module_idx = frame.guest_index;
                    let (module_idx, function) = self
                        .function(module_idx, *func_idx)
                        .ok_or_else(|| anyhow::anyhow!("missing function"))?;

                    let Type(ResultType(params), ResultType(results)) = self
                        .typedef(module_idx, *function.typeidx())
                        .ok_or_else(|| anyhow::anyhow!("missing typedef"))?;

                    let code = self.code(module_idx, function.codeidx());

                    locals.begin_call(&mut value_stack, params, &code.0.locals);

                    let new_frame = Frame {
                        #[cfg(test)]
                        name: "Call",
                        pc: 0,
                        return_unwind_count: 0,
                        instrs: &code.0.expr.0,
                        jump_to_start_of_instrs: false,
                        fence: value_stack.fence(),
                        guest_index: module_idx,
                        params: params.iter().copied().collect(),
                        results: results.iter().copied().collect(),
                    };

                    frame.pc += 1;
                    let old_frame = frame;
                    frames.push(old_frame);
                    frame = new_frame;
                    continue;
                }

                Instr::CallIndirect(TypeIdx(decl_type_idx), table_idx) => {
                    let table_instance_idx = self.table(frame.guest_index, *table_idx);
                    let Type(ResultType(decl_params), ResultType(decl_results)) = self
                        .typedef(module_idx, TypeIdx(*decl_type_idx))
                        .ok_or_else(|| anyhow::anyhow!("missing typedef"))?;

                    let table = &resources.table_instances[table_instance_idx];

                    let idx = value_stack.pop::<i32>();

                    if idx as usize >= table.len() {
                        anyhow::bail!("undefined element {idx}");
                    }

                    let Some(v) = table[idx as u32] else {
                        anyhow::bail!("uninitialized element {idx}");
                    };

                    let v = v.get();
                    let module_idx = (v >> 32) as u32 - 1;
                    let function_idx = v as u32;

                    let (module_idx, function) = self
                        .function(module_idx as usize, FuncIdx(function_idx))
                        .ok_or_else(|| anyhow::anyhow!("missing function"))?;

                    let TypeIdx(func_type_idx) = *function.typeidx();
                    let Type(ResultType(params), ResultType(results)) = self
                        .typedef(module_idx, TypeIdx(func_type_idx))
                        .ok_or_else(|| anyhow::anyhow!("missing typedef"))?;

                    #[cfg(any())]
                    eprintln!("decl({decl_type_idx}) = ({decl_params:?}) -> ({decl_results:?}); real({func_type_idx}) = ({params:?}) -> ({results:?})");

                    if func_type_idx != *decl_type_idx
                        && (decl_params != params || decl_results != results)
                    {
                        anyhow::bail!("indirect call type mismatch");
                    }
                    let code = self.code(module_idx, function.codeidx());

                    locals.begin_call(&mut value_stack, params, &code.0.locals);

                    let new_frame = Frame {
                        #[cfg(test)]
                        name: "Call",
                        pc: 0,
                        return_unwind_count: 0,
                        instrs: &code.0.expr.0,
                        jump_to_start_of_instrs: false,
                        fence: value_stack.fence(),
                        guest_index: module_idx,
                        params: params.iter().copied().collect(),
                        results: results.iter().copied().collect(),
                    };

                    frame.pc += 1;
                    let old_frame = frame;
                    frames.push(old_frame);
                    frame = new_frame;
                    continue;
                }

                Instr::RefNull(_ref_type) => value_stack.push(None),
                Instr::RefIsNull => {
                    let v = value_stack.pop::<RefValue>();

                    value_stack.push::<i32>(if v.is_none() { 1 } else { 0 });
                }

                Instr::RefFunc(FuncIdx(func_idx)) => value_stack.push(NonZeroU64::new(
                    (frame.guest_index as u64 + 1) << 32 | (*func_idx as u64),
                )),

                Instr::DropEmpty => {
                    // XXX(chrisdickinson): we don't know FOR SURE that these are 4-byte values
                    // being dropped. Validation should turn this instruction into a typed Drop.
                    value_stack.pop::<i32>();
                }

                Instr::Drop(val_type) => {
                    value_stack.pop_valtype(*val_type);
                }

                // need to annotate with type!!
                Instr::SelectEmpty => {
                    // XXX(chrisdickinson): we don't know FOR SURE that these are 4-byte values
                    // being selected. The spec says "numeric type" which includes i64/f64 as well!
                    //
                    // That said, the validator rewrites all incoming SelectEmpty instructions to
                    // Select(operands) so the chances of hitting this path are pretty low.
                    let test = value_stack.pop::<i32>();
                    let val1 = value_stack.pop::<i32>();
                    let val2 = value_stack.pop::<i32>();

                    value_stack.push(if test != 0 { val2 } else { val1 })
                }

                Instr::Select(operands) => {
                    let test = value_stack.pop::<i32>();
                    let val1 = value_stack.pop_valtype(operands[0]);
                    let val2 = value_stack.pop_valtype(operands[0]);

                    value_stack.push_value(if test != 0 { val2 } else { val1 })
                }

                Instr::LocalGet(LocalIdx(idx)) => {
                    let value = locals.get(&value_stack, *idx);
                    value_stack.push_value(value);
                }

                Instr::LocalSet(LocalIdx(idx)) => {
                    locals.set(&mut value_stack, *idx);
                }

                Instr::LocalTee(LocalIdx(idx)) => {
                    locals.tee(&mut value_stack, *idx);
                }

                Instr::GlobalGet(global_idx) => {
                    let global_value_idx = self.global(frame.guest_index, *global_idx);

                    let value = unsafe { resources.global_values.get_unchecked(global_value_idx) };

                    value_stack.push_value(value.clone());
                }

                Instr::GlobalSet(global_idx) => {
                    let global_value_idx = self.global(frame.guest_index, *global_idx);
                    let value =
                        unsafe { resources.global_values.get_unchecked_mut(global_value_idx) };

                    match value {
                        StackValue::I32(value) => {
                            *value = value_stack.pop();
                        }
                        StackValue::I64(value) => {
                            *value = value_stack.pop();
                        }
                        StackValue::F32(value) => {
                            *value = value_stack.pop();
                        }
                        StackValue::F64(value) => {
                            *value = value_stack.pop();
                        }
                        StackValue::V128(value) => {
                            let value = value.as_mut();
                            *value = value_stack.pop::<i128>();
                        }
                        StackValue::RefFunc(value) => {
                            *value = value_stack.pop();
                        }
                        StackValue::RefExtern(value) => {
                            *value = value_stack.pop();
                        }
                    }
                }

                Instr::TableGet(table_idx) => {
                    let table_idx = self.table(frame.guest_index, *table_idx);
                    let offset = value_stack.pop::<i32>() as u32;

                    let table = &resources.table_instances[table_idx];
                    if offset >= table.len() as u32 {
                        anyhow::bail!("out of bounds table access");
                    }
                    value_stack.push(table[offset]);
                }

                Instr::TableSet(table_idx) => {
                    let table_idx = self.table(frame.guest_index, *table_idx);
                    let value = value_stack.pop::<RefValue>();
                    let offset = value_stack.pop::<i32>() as u32;

                    let table = &mut resources.table_instances[table_idx];
                    if offset >= table.len() as u32 {
                        anyhow::bail!("out of bounds table access");
                    }

                    table[offset] = value;
                }

                Instr::TableInit(ElemIdx(elem_idx), table_idx) => {
                    let guest_index = frame.guest_index;

                    let Some(elem) = self
                        .elements
                        .get(guest_index)
                        .and_then(|xs| xs.get(*elem_idx as usize))
                    else {
                        anyhow::bail!("element idx out of range");
                    };

                    let table_idx = self.table(frame.guest_index, *table_idx);

                    let count = value_stack.pop::<i32>() as usize;
                    let srcaddr = value_stack.pop::<i32>() as usize;
                    let destaddr = value_stack.pop::<i32>() as usize;

                    let elem_len = if resources
                        .dropped_elements
                        .contains(&(frame.guest_index, *elem_idx as usize))
                    {
                        0
                    } else {
                        elem.len()
                    };

                    if count.saturating_add(srcaddr) > elem_len {
                        anyhow::bail!("out of bounds table access");
                    }
                    if srcaddr > elem_len {
                        anyhow::bail!("out of bounds table access");
                    }

                    let v: Box<[StackValue]> = (srcaddr..srcaddr + count)
                        .map(|idx| elem.value(idx, guest_index, self, &mut *resources))
                        .collect::<anyhow::Result<_>>()?;

                    let table = &mut resources.table_instances[table_idx];
                    if (destaddr + count) > table.len() {
                        anyhow::bail!("out of bounds table access");
                    }
                    if destaddr > table.len() {
                        anyhow::bail!("out of bounds table access");
                    }

                    for (src_idx, dst_idx) in (0..count).zip(destaddr..destaddr + count) {
                        table[dst_idx as u32] = match v[src_idx] {
                            StackValue::RefFunc(xs) => xs,
                            StackValue::RefExtern(xs) => xs,
                            _ => continue,
                        };
                    }
                }

                Instr::ElemDrop(elem_idx) => {
                    if self.elements[frame.guest_index]
                        .get(elem_idx.0 as usize)
                        .is_none()
                    {
                        anyhow::bail!("could not fetch element segement by that id")
                    }

                    resources
                        .dropped_elements
                        .insert((frame.guest_index, elem_idx.0 as usize));
                }

                Instr::TableCopy(to_table_idx, from_table_idx) => {
                    let from_table_idx = self.table(frame.guest_index, *from_table_idx);

                    let to_table_idx = self.table(frame.guest_index, *to_table_idx);

                    let count = value_stack.pop::<i32>() as usize;
                    let srcaddr = value_stack.pop::<i32>() as usize;
                    let destaddr = value_stack.pop::<i32>() as usize;

                    if srcaddr > resources.table_instances[from_table_idx].len() {
                        anyhow::bail!("out of bounds table access");
                    }
                    if destaddr > resources.table_instances[to_table_idx].len() {
                        anyhow::bail!("out of bounds table access");
                    }
                    if destaddr.saturating_add(count)
                        > resources.table_instances[to_table_idx].len()
                    {
                        anyhow::bail!("out of bounds table access");
                    }
                    if srcaddr.saturating_add(count)
                        > resources.table_instances[from_table_idx].len()
                    {
                        anyhow::bail!("out of bounds table access");
                    }

                    let values = (srcaddr..srcaddr + count)
                        .map(|src_idx| resources.table_instances[from_table_idx][src_idx as u32])
                        .collect::<Box<[_]>>();

                    for (src_idx, dst_idx) in (0..count).zip(destaddr..destaddr + count) {
                        resources.table_instances[to_table_idx][dst_idx as u32] = values[src_idx];
                    }
                }

                Instr::TableGrow(table_idx) => {
                    let table_idx = self.table(frame.guest_index, *table_idx);
                    let size = value_stack.pop::<i32>();
                    let fill = value_stack.pop::<RefValue>();

                    // TODO: respect upper bound size of table if present?
                    let tbl = &mut resources.table_instances[table_idx];
                    let old_len = tbl.grow(size, fill);

                    value_stack.push::<i32>(old_len);
                }

                Instr::TableSize(table_idx) => {
                    let table_idx = self.table(frame.guest_index, *table_idx);
                    value_stack.push(resources.table_instances[table_idx].len() as i32);
                }

                Instr::TableFill(table_idx) => {
                    let table_idx = self.table(frame.guest_index, *table_idx);
                    let size = value_stack.pop::<i32>();
                    let fill = value_stack.pop::<RefValue>();
                    let offset = value_stack.pop::<i32>();

                    // TODO: respect upper bound size of table if present?
                    let tbl = &mut resources.table_instances[table_idx];

                    let offset = offset as usize;
                    let size = size as usize;
                    if offset > tbl.len() || offset + size > tbl.len() {
                        anyhow::bail!("out of bounds table access");
                    }

                    tbl[offset..offset + size].fill(fill);
                }

                Instr::I32Load(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<u32>();
                    let offset = mem.offset().saturating_add(v as usize);
                    let arr = memory_region.load(offset)?;
                    value_stack.push::<i32>(i32::from_le_bytes(arr));
                }

                Instr::I64Load(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<u32>();
                    let offset = mem.offset().saturating_add(v as usize);
                    let arr = memory_region.load(offset)?;
                    value_stack.push::<i64>(i64::from_le_bytes(arr));
                }

                Instr::F32Load(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<u32>();
                    let offset = mem.offset().saturating_add(v as usize);
                    let arr = memory_region.load(offset)?;
                    value_stack.push::<f32>(f32::from_le_bytes(arr));
                }

                Instr::F64Load(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<u32>();
                    let offset = mem.offset().saturating_add(v as usize);
                    let arr = memory_region.load(offset)?;
                    value_stack.push::<f64>(f64::from_le_bytes(arr));
                }

                Instr::I32Load8S(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<u32>();
                    let offset = mem.offset().saturating_add(v as usize);
                    let arr = memory_region.load::<1>(offset)?;
                    value_stack.push((arr[0] as i8) as i32);
                }

                Instr::I32Load8U(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<u32>();
                    let offset = mem.offset().saturating_add(v as usize);
                    let arr = memory_region.load::<1>(offset)?;
                    value_stack.push(arr[0] as u32 as i32);
                }

                Instr::I32Load16S(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<u32>();
                    let offset = mem.offset().saturating_add(v as usize);
                    let arr = memory_region.load(offset)?;
                    value_stack.push::<i32>(i16::from_le_bytes(arr) as i32);
                }

                Instr::I32Load16U(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<u32>();
                    let offset = mem.offset().saturating_add(v as usize);
                    let arr = memory_region.load(offset)?;
                    value_stack.push::<i32>(u16::from_le_bytes(arr) as i32);
                }

                Instr::I64Load8S(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<u32>();
                    let offset = mem.offset().saturating_add(v as usize);
                    let arr = memory_region.load::<1>(offset)?;
                    value_stack.push((arr[0] as i8) as i64);
                }

                Instr::I64Load8U(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<u32>();
                    let offset = mem.offset().saturating_add(v as usize);
                    let arr = memory_region.load::<1>(offset)?;
                    value_stack.push(arr[0] as u64 as i64);
                }

                Instr::I64Load16S(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<u32>();
                    let offset = mem.offset().saturating_add(v as usize);
                    let arr = memory_region.load(offset)?;
                    value_stack.push::<i64>(i16::from_le_bytes(arr) as i64);
                }

                Instr::I64Load16U(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<u32>();
                    let offset = mem.offset().saturating_add(v as usize);
                    let arr = memory_region.load(offset)?;
                    value_stack.push::<i64>(u16::from_le_bytes(arr) as i64);
                }

                Instr::I64Load32S(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<u32>();
                    let offset = mem.offset().saturating_add(v as usize);
                    let arr = memory_region.load(offset)?;
                    value_stack.push::<i64>(i32::from_le_bytes(arr) as i64);
                }

                Instr::I64Load32U(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<u32>();
                    let offset = mem.offset().saturating_add(v as usize);
                    let arr = memory_region.load(offset)?;
                    value_stack.push::<i64>(u32::from_le_bytes(arr) as i64);
                }

                Instr::I32Store(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &mut resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<i32>();
                    let t = value_stack.pop::<i32>();

                    let offset = mem.offset().saturating_add(t as usize);
                    memory_region.store(offset, &v.to_le_bytes())?;
                }

                Instr::I64Store(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &mut resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<i64>();
                    let t = value_stack.pop::<i32>();

                    let offset = mem.offset().saturating_add(t as usize);
                    memory_region.store(offset, &v.to_le_bytes())?;
                }

                Instr::F32Store(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &mut resources.memory_regions[memory_idx];

                    let v = f32::from_bits(value_stack.pop::<u32>());
                    let t = value_stack.pop::<i32>();

                    let offset = mem.offset().saturating_add(t as usize);
                    memory_region.store(offset, &v.to_le_bytes())?;
                }

                Instr::F64Store(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &mut resources.memory_regions[memory_idx];

                    let v = f64::from_bits(value_stack.pop::<u64>());
                    let t = value_stack.pop::<i32>();

                    let offset = mem.offset().saturating_add(t as usize);
                    memory_region.store(offset, &v.to_le_bytes())?;
                }

                Instr::I32Store8(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &mut resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<i32>();
                    let t = value_stack.pop::<i32>();

                    let v = v as i8;
                    let offset = mem.offset().saturating_add(t as usize);
                    memory_region.store(offset, &[v as u8])?;
                }

                Instr::I32Store16(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &mut resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<i32>();
                    let t = value_stack.pop::<i32>();

                    let v = v as i16;
                    let offset = mem.offset().saturating_add(t as usize);
                    memory_region.store(offset, &v.to_le_bytes())?;
                }

                Instr::I64Store8(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &mut resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<i64>();
                    let t = value_stack.pop::<i32>();

                    let v = v as i8;
                    let offset = mem.offset().saturating_add(t as usize);
                    memory_region.store(offset, &[v as u8])?;
                }

                Instr::I64Store16(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &mut resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<i64>();
                    let t = value_stack.pop::<i32>();

                    let v = v as i16;
                    let offset = mem.offset().saturating_add(t as usize);
                    memory_region.store(offset, &v.to_le_bytes())?;
                }

                Instr::I64Store32(mem) => {
                    let memory_idx = self.memory(frame.guest_index, MemIdx(mem.memidx() as u32));
                    let memory_region = &mut resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<i64>();
                    let t = value_stack.pop::<i32>();

                    let v = v as i32;
                    let offset = mem.offset().saturating_add(t as usize);
                    memory_region.store(offset, &v.to_le_bytes())?;
                }

                Instr::MemorySize(mem_idx) => {
                    let memory_idx = self.memory(frame.guest_index, *mem_idx);
                    let memory_region = &mut resources.memory_regions[memory_idx];
                    value_stack.push::<i32>(memory_region.page_count() as i32);
                }

                Instr::MemoryGrow(mem_idx) => {
                    let memory_idx = self.memory(frame.guest_index, *mem_idx);
                    let memory_region = &mut resources.memory_regions[memory_idx];

                    let v = value_stack.pop::<i32>();

                    value_stack.push(if let Ok(page_count) = memory_region.grow(v as usize) {
                        page_count as i32
                    } else {
                        -1i32
                    });
                }

                Instr::MemoryInit(data_idx, mem_idx) => {
                    let Some(data) = self.data[frame.guest_index].get(data_idx.0 as usize) else {
                        anyhow::bail!("could not fetch data by that id")
                    };
                    let data = if resources
                        .dropped_data
                        .contains(&(frame.guest_index, data_idx.0 as usize))
                    {
                        &[]
                    } else {
                        match data {
                            Data::Active(ByteVec(v), _, _) => &**v,
                            Data::Passive(ByteVec(v)) => &**v,
                        }
                    };

                    let memory_idx = self.memory(frame.guest_index, *mem_idx);

                    let memory_region = &mut resources.memory_regions[memory_idx];
                    let count = value_stack.pop::<i32>() as usize;
                    let srcaddr = value_stack.pop::<i32>() as usize;
                    let destaddr = value_stack.pop::<i32>() as usize;

                    if srcaddr > data.len() {
                        anyhow::bail!("out of bounds memory access");
                    }
                    if (srcaddr + count) > data.len() {
                        anyhow::bail!("out of bounds memory access");
                    }
                    let from_slice = &data[srcaddr..(srcaddr + count)];

                    if destaddr > memory_region.len() {
                        anyhow::bail!("out of bounds memory access");
                    }

                    if (destaddr + count) > memory_region.len() {
                        anyhow::bail!("out of bounds memory access");
                    }

                    memory_region.copy_data(from_slice, destaddr);
                }

                Instr::DataDrop(data_idx) => {
                    if self.data[frame.guest_index]
                        .get(data_idx.0 as usize)
                        .is_none()
                    {
                        anyhow::bail!("could not fetch data by that id")
                    }

                    resources
                        .dropped_data
                        .insert((frame.guest_index, data_idx.0 as usize));
                }

                Instr::MemoryCopy(from_mem_idx, to_mem_idx) => {
                    let from_memory_idx = self.memory(frame.guest_index, *from_mem_idx);

                    let to_memory_idx = self.memory(frame.guest_index, *to_mem_idx);

                    // if these are the same memory, we're going to borrow it once mutably.
                    if from_memory_idx == to_memory_idx {
                        let memory_region = &mut resources.memory_regions[from_memory_idx];
                        let count = value_stack.pop::<i32>() as usize;
                        let srcaddr = value_stack.pop::<i32>() as usize;
                        let destaddr = value_stack.pop::<i32>() as usize;

                        if srcaddr > memory_region.len() {
                            anyhow::bail!("out of bounds memory access");
                        }
                        if destaddr > memory_region.len() {
                            anyhow::bail!("out of bounds memory access");
                        }
                        if destaddr.saturating_add(count) > memory_region.len() {
                            anyhow::bail!("out of bounds memory access");
                        }
                        if srcaddr.saturating_add(count) > memory_region.len() {
                            anyhow::bail!("out of bounds memory access");
                        }
                        memory_region.copy_overlapping_data(destaddr, srcaddr, count);
                    } else {
                        let _from_memory_region = &resources.memory_regions[from_memory_idx];
                        let _to_memory_region = &resources.memory_regions[to_memory_idx];
                        // memory_region.copy_data(from_slice, count);
                        todo!()
                    }
                }

                Instr::MemoryFill(mem_idx) => {
                    let memory_idx = self.memory(frame.guest_index, *mem_idx);

                    let memory_region = &mut resources.memory_regions[memory_idx];

                    let count = value_stack.pop::<i32>() as usize;
                    let val = value_stack.pop::<i32>() as usize;
                    let offset = value_stack.pop::<i32>() as usize;

                    memory_region.fill_data(val as u8, offset, count)?;
                }

                Instr::I32Const(v) => {
                    value_stack.push(*v);
                }

                Instr::I64Const(v) => {
                    value_stack.push(*v);
                }

                Instr::F32Const(v) => {
                    value_stack.push(*v);
                }

                Instr::F64Const(v) => {
                    value_stack.push(*v);
                }

                Instr::I32Eqz => {
                    let value = value_stack.pop::<i32>() as usize;
                    value_stack.push::<i32>(if value == 0 { 1 } else { 0 });
                }

                Instr::I32Eq => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    value_stack.push::<i32>(if lhs == rhs { 1 } else { 0 });
                }

                Instr::I32Ne => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    value_stack.push(if lhs != rhs { 1 } else { 0 });
                }

                Instr::I32LtS => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    value_stack.push::<i32>(if lhs < rhs { 1 } else { 0 });
                }

                Instr::I32LtU => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    value_stack.push::<i32>(if (lhs as u32) < (rhs as u32) { 1 } else { 0 });
                }

                Instr::I32GtS => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    value_stack.push::<i32>(if lhs > rhs { 1 } else { 0 });
                }

                Instr::I32GtU => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    value_stack.push::<i32>(if (lhs as u32) > (rhs as u32) { 1 } else { 0 });
                }

                Instr::I32LeS => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    value_stack.push::<i32>(if lhs <= rhs { 1 } else { 0 });
                }

                Instr::I32LeU => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    value_stack.push::<i32>(if (lhs as u32) <= (rhs as u32) { 1 } else { 0 });
                }

                Instr::I32GeS => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    value_stack.push::<i32>(if lhs >= rhs { 1 } else { 0 });
                }

                Instr::I32GeU => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    value_stack.push::<i32>(if (lhs as u32) >= (rhs as u32) { 1 } else { 0 });
                }

                Instr::I64Eqz => {
                    let v = value_stack.pop::<i64>();
                    value_stack.push::<i32>(if v == 0 { 1 } else { 0 });
                }

                Instr::I64Eq => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    value_stack.push::<i32>(if lhs == rhs { 1 } else { 0 });
                }

                Instr::I64Ne => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    value_stack.push(if lhs != rhs { 1 } else { 0 });
                }

                Instr::I64LtS => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    value_stack.push::<i32>(if lhs < rhs { 1 } else { 0 });
                }

                Instr::I64LtU => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    value_stack.push::<i32>(if (lhs as u64) < (rhs as u64) { 1 } else { 0 });
                }

                Instr::I64GtS => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    value_stack.push::<i32>(if lhs > rhs { 1 } else { 0 });
                }

                Instr::I64GtU => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    value_stack.push::<i32>(if (lhs as u64) > (rhs as u64) { 1 } else { 0 });
                }

                Instr::I64LeS => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    value_stack.push::<i32>(if lhs <= rhs { 1 } else { 0 });
                }

                Instr::I64LeU => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    value_stack.push::<i32>(if (lhs as u64) <= (rhs as u64) { 1 } else { 0 });
                }

                Instr::I64GeS => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    value_stack.push::<i32>(if lhs >= rhs { 1 } else { 0 });
                }

                Instr::I64GeU => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    value_stack.push::<i32>(if (lhs as u64) >= (rhs as u64) { 1 } else { 0 });
                }

                Instr::F32Eq => {
                    let rhs = f32::from_bits(value_stack.pop::<u32>());
                    let lhs = f32::from_bits(value_stack.pop::<u32>());
                    value_stack.push::<i32>(if lhs == rhs { 1 } else { 0 });
                }

                Instr::F32Ne => {
                    let rhs = f32::from_bits(value_stack.pop::<u32>());
                    let lhs = f32::from_bits(value_stack.pop::<u32>());
                    value_stack.push(if lhs != rhs { 1 } else { 0 });
                }

                Instr::F32Lt => {
                    let rhs = f32::from_bits(value_stack.pop::<u32>());
                    let lhs = f32::from_bits(value_stack.pop::<u32>());
                    value_stack.push::<i32>(if lhs < rhs { 1 } else { 0 });
                }

                Instr::F32Gt => {
                    let rhs = f32::from_bits(value_stack.pop::<u32>());
                    let lhs = f32::from_bits(value_stack.pop::<u32>());
                    value_stack.push::<i32>(if lhs > rhs { 1 } else { 0 });
                }

                Instr::F32Le => {
                    let rhs = f32::from_bits(value_stack.pop::<u32>());
                    let lhs = f32::from_bits(value_stack.pop::<u32>());
                    value_stack.push::<i32>(if lhs <= rhs { 1 } else { 0 });
                }

                Instr::F32Ge => {
                    let rhs = f32::from_bits(value_stack.pop::<u32>());
                    let lhs = f32::from_bits(value_stack.pop::<u32>());
                    value_stack.push::<i32>(if lhs >= rhs { 1 } else { 0 });
                }

                Instr::F64Eq => {
                    let rhs = f64::from_bits(value_stack.pop::<u64>());
                    let lhs = f64::from_bits(value_stack.pop::<u64>());
                    value_stack.push::<i32>(if lhs == rhs { 1 } else { 0 });
                }

                Instr::F64Ne => {
                    let rhs = f64::from_bits(value_stack.pop::<u64>());
                    let lhs = f64::from_bits(value_stack.pop::<u64>());
                    value_stack.push(if lhs != rhs { 1 } else { 0 });
                }

                Instr::F64Lt => {
                    let rhs = f64::from_bits(value_stack.pop::<u64>());
                    let lhs = f64::from_bits(value_stack.pop::<u64>());
                    value_stack.push::<i32>(if lhs < rhs { 1 } else { 0 });
                }

                Instr::F64Gt => {
                    let rhs = f64::from_bits(value_stack.pop::<u64>());
                    let lhs = f64::from_bits(value_stack.pop::<u64>());
                    value_stack.push::<i32>(if lhs > rhs { 1 } else { 0 });
                }

                Instr::F64Le => {
                    let rhs = f64::from_bits(value_stack.pop::<u64>());
                    let lhs = f64::from_bits(value_stack.pop::<u64>());
                    value_stack.push::<i32>(if lhs <= rhs { 1 } else { 0 });
                }

                Instr::F64Ge => {
                    let rhs = f64::from_bits(value_stack.pop::<u64>());
                    let lhs = f64::from_bits(value_stack.pop::<u64>());
                    value_stack.push::<i32>(if lhs >= rhs { 1 } else { 0 });
                }

                Instr::I32Clz => {
                    let op = value_stack.pop::<i32>();
                    value_stack.push::<i32>(op.leading_zeros() as i32);
                }

                Instr::I32Ctz => {
                    let op = value_stack.pop::<i32>();
                    value_stack.push::<i32>(op.trailing_zeros() as i32);
                }

                Instr::I32Popcnt => {
                    let op = value_stack.pop::<i32>();
                    value_stack.push::<i32>(op.count_ones() as i32);
                }

                Instr::I32Add => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    value_stack.push::<i32>(lhs.wrapping_add(rhs));
                }

                Instr::I32Sub => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    value_stack.push::<i32>(lhs.wrapping_sub(rhs));
                }

                Instr::I32Mul => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    value_stack.push::<i32>(lhs.wrapping_mul(rhs));
                }

                Instr::I32DivS => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();

                    if rhs == 0 {
                        anyhow::bail!("i32.divS: integer divide by zero");
                    }

                    let Some(result) = lhs.checked_div(rhs) else {
                        anyhow::bail!("i32.divS: integer overflow");
                    };

                    value_stack.push::<i32>(result);
                }

                Instr::I32DivU => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();

                    let lhs = lhs as u32;
                    let rhs = rhs as u32;

                    if rhs == 0 {
                        anyhow::bail!("i32.divU: integer divide by zero");
                    }

                    let Some(result) = lhs.checked_div(rhs) else {
                        anyhow::bail!("i32.divU: integer overflow");
                    };

                    value_stack.push::<i32>(result as i32);
                }

                Instr::I32RemS => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();

                    if rhs == 0 {
                        anyhow::bail!("i32.remS: integer divide by zero");
                    }

                    let result = lhs.wrapping_rem(rhs);

                    value_stack.push::<i32>(result);
                }

                Instr::I32RemU => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();

                    let lhs = lhs as u32;
                    let rhs = rhs as u32;

                    if rhs == 0 {
                        anyhow::bail!("i32.rem_u: integer divide by zero");
                    }

                    let result = lhs.wrapping_rem(rhs);

                    value_stack.push::<i32>(result as i32);
                }

                Instr::I32And => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    value_stack.push(lhs & rhs);
                }

                Instr::I32Ior => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    value_stack.push(lhs | rhs);
                }

                Instr::I32Xor => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    value_stack.push(lhs ^ rhs);
                }

                Instr::I32Shl => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    value_stack.push::<i32>(lhs.wrapping_shl(rhs as u32));
                }

                Instr::I32ShrS => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    value_stack.push::<i32>(lhs.wrapping_shr(rhs as u32));
                }

                Instr::I32ShrU => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    let lhs = lhs as u32;
                    value_stack.push::<i32>(lhs.wrapping_shr(rhs as u32) as i32);
                }

                Instr::I32Rol => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    value_stack.push::<i32>(lhs.rotate_left(rhs as u32));
                }

                Instr::I32Ror => {
                    let rhs = value_stack.pop::<i32>();
                    let lhs = value_stack.pop::<i32>();
                    value_stack.push::<i32>(lhs.rotate_right(rhs as u32));
                }

                Instr::I64Clz => {
                    let op = value_stack.pop::<i64>();
                    value_stack.push::<i64>(op.leading_zeros() as i64);
                }

                Instr::I64Ctz => {
                    let op = value_stack.pop::<i64>();
                    value_stack.push::<i64>(op.trailing_zeros() as i64);
                }

                Instr::I64Popcnt => {
                    let op = value_stack.pop::<i64>();
                    value_stack.push::<i64>(op.count_ones() as i64);
                }

                Instr::I64Add => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    value_stack.push::<i64>(lhs.wrapping_add(rhs));
                }

                Instr::I64Sub => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    value_stack.push::<i64>(lhs.wrapping_sub(rhs));
                }

                Instr::I64Mul => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    value_stack.push::<i64>(lhs.wrapping_mul(rhs));
                }

                Instr::I64DivS => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();

                    if rhs == 0 {
                        anyhow::bail!("i64.divS: integer divide by zero");
                    }

                    let Some(result) = lhs.checked_div(rhs) else {
                        anyhow::bail!("i32.divS: integer overflow");
                    };

                    value_stack.push::<i64>(result);
                }

                Instr::I64DivU => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();

                    let lhs = lhs as u64;
                    let rhs = rhs as u64;

                    if rhs == 0 {
                        anyhow::bail!("i64.divU: integer divide by zero");
                    }

                    let Some(result) = lhs.checked_div(rhs) else {
                        anyhow::bail!("i64.divU: integer overflow");
                    };

                    value_stack.push::<i64>(result as i64);
                }

                Instr::I64RemS => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    if rhs == 0 {
                        anyhow::bail!("i64.remS: integer divide by zero");
                    }
                    let result = lhs.wrapping_rem(rhs);
                    value_stack.push::<i64>(result);
                }

                Instr::I64RemU => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    if rhs == 0 {
                        anyhow::bail!("i64.remS: integer divide by zero");
                    }
                    let lhs = lhs as u64;
                    let rhs = rhs as u64;

                    let result = lhs.wrapping_rem(rhs);
                    value_stack.push::<i64>(result as i64);
                }

                Instr::I64And => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    value_stack.push(lhs & rhs);
                }

                Instr::I64Ior => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    value_stack.push(lhs | rhs);
                }

                Instr::I64Xor => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    value_stack.push(lhs ^ rhs);
                }

                Instr::I64Shl => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();

                    value_stack.push::<i64>(lhs.wrapping_shl(rhs as u32));
                }

                Instr::I64ShrS => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    value_stack.push::<i64>(lhs.wrapping_shr(rhs as u32));
                }

                Instr::I64ShrU => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    let lhs = lhs as u64;
                    value_stack.push::<i64>(lhs.wrapping_shr(rhs as u32) as i64);
                }

                Instr::I64Rol => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    value_stack.push::<i64>(lhs.rotate_left(rhs as u32));
                }

                Instr::I64Ror => {
                    let rhs = value_stack.pop::<i64>();
                    let lhs = value_stack.pop::<i64>();
                    value_stack.push::<i64>(lhs.rotate_right(rhs as u32));
                }

                Instr::F32Abs => {
                    let op = f32::from_bits(value_stack.pop::<u32>());
                    value_stack.push::<f32>(op.abs());
                }

                Instr::F32Neg => {
                    let op = f32::from_bits(value_stack.pop::<u32>());
                    value_stack.push(-op);
                }

                Instr::F32Ceil => {
                    let op = f32::from_bits(value_stack.pop::<u32>());
                    value_stack.push::<f32>(op.ceil());
                }

                Instr::F32Floor => {
                    let op = f32::from_bits(value_stack.pop::<u32>());
                    value_stack.push::<f32>(op.floor());
                }

                Instr::F32Trunc => {
                    let op = f32::from_bits(value_stack.pop::<u32>());
                    value_stack.push::<f32>(op.trunc());
                }

                Instr::F32NearestInt => {
                    let op = f32::from_bits(value_stack.pop::<u32>());
                    value_stack.push::<f32>(nearestf32(op));
                }

                Instr::F32Sqrt => {
                    let op = f32::from_bits(value_stack.pop::<u32>());
                    value_stack.push::<f32>(op.sqrt());
                }

                Instr::F32Add => {
                    let rhs = value_stack.pop::<u32>();
                    let lhs = value_stack.pop::<u32>();

                    let rhs = f32::from_bits(rhs);
                    let lhs = f32::from_bits(lhs);

                    value_stack.push(lhs + rhs);
                }

                Instr::F32Sub => {
                    let rhs = f32::from_bits(value_stack.pop::<u32>());
                    let lhs = f32::from_bits(value_stack.pop::<u32>());
                    value_stack.push(lhs - rhs);
                }

                Instr::F32Mul => {
                    let rhs = f32::from_bits(value_stack.pop::<u32>());
                    let lhs = f32::from_bits(value_stack.pop::<u32>());
                    value_stack.push(lhs * rhs);
                }

                Instr::F32Div => {
                    let rhs = f32::from_bits(value_stack.pop::<u32>());
                    let lhs = f32::from_bits(value_stack.pop::<u32>());
                    value_stack.push(lhs / rhs);
                }

                Instr::F32Min => {
                    let rhs = value_stack.pop::<u32>();
                    let lhs = value_stack.pop::<u32>();

                    if (lhs & 0x7fc0_0000) == 0x7fc0_0000 {
                        value_stack.push::<u32>(lhs);
                    } else if (rhs & 0x7fc0_0000) == 0x7fc0_0000 {
                        value_stack.push::<u32>(rhs);
                    } else {
                        let rhs = f32::from_bits(rhs);
                        let lhs = f32::from_bits(lhs);
                        value_stack.push::<f32>(lhs.min(rhs));
                    }
                }

                Instr::F32Max => {
                    let rhs = value_stack.pop::<u32>();
                    let lhs = value_stack.pop::<u32>();

                    if (lhs & 0x7fc0_0000) == 0x7fc0_0000 {
                        value_stack.push::<u32>(lhs);
                    } else if (rhs & 0x7fc0_0000) == 0x7fc0_0000 {
                        value_stack.push::<u32>(rhs);
                    } else {
                        let rhs = f32::from_bits(rhs);
                        let lhs = f32::from_bits(lhs);
                        value_stack.push::<f32>(lhs.max(rhs));
                    }
                }

                Instr::F32CopySign => {
                    let rhs = f32::from_bits(value_stack.pop::<u32>());
                    let lhs = f32::from_bits(value_stack.pop::<u32>());
                    value_stack.push::<f32>(lhs.copysign(rhs));
                }

                Instr::F64Abs => {
                    let op = f64::from_bits(value_stack.pop::<u64>());
                    value_stack.push::<f64>(op.abs());
                }

                Instr::F64Neg => {
                    let op = f64::from_bits(value_stack.pop::<u64>());
                    value_stack.push(-op);
                }

                Instr::F64Ceil => {
                    let op = f64::from_bits(value_stack.pop::<u64>());
                    value_stack.push::<f64>(op.ceil());
                }

                Instr::F64Floor => {
                    let op = f64::from_bits(value_stack.pop::<u64>());
                    value_stack.push::<f64>(op.floor());
                }

                Instr::F64Trunc => {
                    let op = f64::from_bits(value_stack.pop::<u64>());
                    value_stack.push::<f64>(op.trunc());
                }

                Instr::F64NearestInt => {
                    let op = f64::from_bits(value_stack.pop::<u64>());
                    value_stack.push::<f64>(nearestf64(op));
                }

                Instr::F64Sqrt => {
                    let op = f64::from_bits(value_stack.pop::<u64>());
                    value_stack.push::<f64>(op.sqrt());
                }

                Instr::F64Add => {
                    let rhs = f64::from_bits(value_stack.pop::<u64>());
                    let lhs = f64::from_bits(value_stack.pop::<u64>());
                    value_stack.push(lhs + rhs);
                }

                Instr::F64Sub => {
                    let rhs = f64::from_bits(value_stack.pop::<u64>());
                    let lhs = f64::from_bits(value_stack.pop::<u64>());
                    value_stack.push(lhs - rhs);
                }

                Instr::F64Mul => {
                    let rhs = f64::from_bits(value_stack.pop::<u64>());
                    let lhs = f64::from_bits(value_stack.pop::<u64>());
                    value_stack.push(lhs * rhs);
                }

                Instr::F64Div => {
                    let rhs = f64::from_bits(value_stack.pop::<u64>());
                    let lhs = f64::from_bits(value_stack.pop::<u64>());
                    value_stack.push(lhs / rhs);
                }

                Instr::F64Min => {
                    let rhs = value_stack.pop::<u64>();
                    let lhs = value_stack.pop::<u64>();

                    if (lhs & 0x7ff8_0000_0000_0000) == 0x7ff8_0000_0000_0000 {
                        value_stack.push::<u64>(lhs);
                    } else if (rhs & 0x7ff8_0000_0000_0000) == 0x7ff8_0000_0000_0000 {
                        value_stack.push::<u64>(rhs);
                    } else {
                        let rhs = f64::from_bits(rhs);
                        let lhs = f64::from_bits(lhs);
                        value_stack.push::<f64>(lhs.min(rhs));
                    }
                }

                Instr::F64Max => {
                    let rhs = value_stack.pop::<u64>();
                    let lhs = value_stack.pop::<u64>();

                    if (lhs & 0x7ff8_0000_0000_0000) == 0x7ff8_0000_0000_0000 {
                        value_stack.push::<u64>(lhs);
                    } else if (rhs & 0x7ff8_0000_0000_0000) == 0x7ff8_0000_0000_0000 {
                        value_stack.push::<u64>(rhs);
                    } else {
                        let rhs = f64::from_bits(rhs);
                        let lhs = f64::from_bits(lhs);
                        value_stack.push::<f64>(lhs.max(rhs));
                    }
                }

                Instr::F64CopySign => {
                    let rhs = f64::from_bits(value_stack.pop::<u64>());
                    let lhs = f64::from_bits(value_stack.pop::<u64>());

                    value_stack.push::<f64>(lhs.copysign(rhs));
                }

                Instr::I32ConvertI64 => {
                    let op = value_stack.pop::<i64>();

                    value_stack.push::<i32>(op as i32);
                }

                Instr::I32SConvertF32 => {
                    let op = f32::from_bits(value_stack.pop::<u32>());

                    if op.is_nan() {
                        anyhow::bail!("i32.trunc_f32_s: invalid conversion to integer");
                    }

                    if op.is_infinite() {
                        anyhow::bail!("i32.trunc_f32_s: integer overflow");
                    }

                    let op = op.trunc();

                    if op >= (i32::MAX as f32) || op < (i32::MIN as f32) {
                        anyhow::bail!("i32.trunc_f32_s: integer overflow");
                    }

                    value_stack.push::<i32>(op as i32);
                }

                Instr::I32UConvertF32 => {
                    let op = f32::from_bits(value_stack.pop::<u32>());

                    if op.is_nan() {
                        anyhow::bail!("i32.trunc_f32_u: invalid conversion to integer");
                    }

                    if op.is_infinite() {
                        anyhow::bail!("i32.trunc_f32_u: integer overflow");
                    }

                    let op = op.trunc();

                    if op >= (u32::MAX as f32) || op < (u32::MIN as f32) {
                        anyhow::bail!("i32.trunc_f32_u: integer overflow");
                    }

                    value_stack.push::<i32>(op as u32 as i32);
                }

                Instr::I32SConvertF64 => {
                    let op = f64::from_bits(value_stack.pop::<u64>());

                    if op.is_nan() {
                        anyhow::bail!("i32.trunc_f64_s: invalid conversion to integer");
                    }

                    if op.is_infinite() {
                        anyhow::bail!("i32.trunc_f64_s: integer overflow");
                    }

                    let op = op as i64;

                    if op > (i32::MAX as i64) || op < (i32::MIN as i64) {
                        anyhow::bail!("i32.trunc_f64_s: integer overflow");
                    }

                    value_stack.push::<i32>(op as i32);
                }

                Instr::I32UConvertF64 => {
                    let op = f64::from_bits(value_stack.pop::<u64>());

                    if op.is_nan() {
                        anyhow::bail!("i32.trunc_f64_u: invalid conversion to integer");
                    }

                    if op.is_infinite() {
                        anyhow::bail!("i32.trunc_f64_u: integer overflow");
                    }

                    let op = op as i64;

                    if op > (u32::MAX as i64) || op < (u32::MIN as i64) {
                        anyhow::bail!("i32.trunc_f64_u: integer overflow");
                    }

                    value_stack.push::<i32>(op as u32 as i32);
                }

                Instr::I64SConvertI32 => {
                    let op = value_stack.pop::<i32>();

                    value_stack.push::<i64>(op as i64);
                }

                Instr::I64UConvertI32 => {
                    let op = value_stack.pop::<i32>();

                    value_stack.push::<i64>(op as u32 as i64);
                }

                Instr::I64SConvertF32 => {
                    let op = f32::from_bits(value_stack.pop::<u32>());

                    if op.is_nan() {
                        anyhow::bail!("i64.trunc_f32_s: invalid conversion to integer");
                    }

                    if op.is_infinite() {
                        anyhow::bail!("i64.trunc_f32_s: integer overflow");
                    }

                    let op = op.trunc();

                    if op >= (i64::MAX as f32) || op < (i64::MIN as f32) {
                        anyhow::bail!("i64.trunc_f32_s: integer overflow");
                    }

                    value_stack.push::<i64>(op as i64);
                }

                Instr::I64UConvertF32 => {
                    let op = f32::from_bits(value_stack.pop::<u32>());

                    if op.is_nan() {
                        anyhow::bail!("i64.trunc_f32_u: invalid conversion to integer");
                    }

                    if op.is_infinite() {
                        anyhow::bail!("i64.trunc_f32_u: integer overflow");
                    }

                    let op = op.trunc();

                    if op >= (u64::MAX as f32) || op < (u64::MIN as f32) {
                        anyhow::bail!("i64.trunc_f32_u: integer overflow");
                    }

                    value_stack.push::<i64>(op as u64 as i64);
                }

                Instr::I64SConvertF64 => {
                    let op = f64::from_bits(value_stack.pop::<u64>());

                    if op.is_nan() {
                        anyhow::bail!("i64.trunc_f64_s: invalid conversion to integer");
                    }

                    if op.is_infinite() {
                        anyhow::bail!("i64.trunc_f64_s: integer overflow");
                    }

                    let op = op.trunc();

                    if op >= (i64::MAX as f64) || op < (i64::MIN as f64) {
                        anyhow::bail!("i64.trunc_f64_s: integer overflow");
                    }

                    value_stack.push::<i64>(op as i64);
                }

                Instr::I64UConvertF64 => {
                    let op = f64::from_bits(value_stack.pop::<u64>());

                    if op.is_nan() {
                        anyhow::bail!("i64.trunc_f64_u: invalid conversion to integer");
                    }

                    if op.is_infinite() {
                        anyhow::bail!("i64.trunc_f64_u: integer overflow");
                    }

                    let op = op.trunc();

                    if op >= (u64::MAX as f64) || op < (u64::MIN as f64) {
                        anyhow::bail!("i64.trunc_f64_u: integer overflow");
                    }

                    value_stack.push::<i64>(op as u64 as i64);
                }

                Instr::F32SConvertI32 => {
                    let op = value_stack.pop::<i32>();

                    value_stack.push::<f32>(op as f32);
                }

                Instr::F32UConvertI32 => {
                    let op = value_stack.pop::<i32>();

                    value_stack.push::<f32>(op as u32 as f32);
                }

                Instr::F32SConvertI64 => {
                    let op = value_stack.pop::<i64>();

                    value_stack.push::<f32>(op as f32);
                }

                Instr::F32UConvertI64 => {
                    let op = value_stack.pop::<i64>();

                    value_stack.push::<f32>(op as u64 as f32);
                }

                Instr::F32ConvertF64 => {
                    let op = f64::from_bits(value_stack.pop::<u64>());

                    let mut op = op as f32;
                    if op.is_nan() {
                        op = f32::from_bits(match op.to_bits() {
                            0x7fe00000 => 0x7fc00000,
                            0xffc00000 => 0x7fc00000,
                            0xffe00000 => 0x7fc00000,
                            passthru => passthru,
                        });
                    }

                    value_stack.push::<u32>(op.to_bits());
                }

                Instr::F64SConvertI32 => {
                    let op = value_stack.pop::<i32>();

                    value_stack.push::<f64>(op as f64);
                }

                Instr::F64UConvertI32 => {
                    let op = value_stack.pop::<i32>();

                    value_stack.push::<f64>(op as u32 as f64);
                }

                Instr::F64SConvertI64 => {
                    let op = value_stack.pop::<i64>();

                    value_stack.push::<f64>(op as f64);
                }

                Instr::F64UConvertI64 => {
                    let op = value_stack.pop::<i64>();

                    value_stack.push::<f64>(op as u64 as f64);
                }

                Instr::F64ConvertF32 => {
                    let op = f32::from_bits(value_stack.pop::<u32>());

                    let mut op = op as f64;
                    if op.is_nan() {
                        op = f64::from_bits(match op.to_bits() {
                            0x7ffc000000000000 => 0x7ff8000000000000,
                            0xfff8000000000000 => 0x7ff8000000000000,
                            0xfffc000000000000 => 0x7ff8000000000000,
                            passthru => passthru,
                        });
                    }

                    value_stack.push::<u64>(op.to_bits());
                }

                Instr::I32ReinterpretF32 => {
                    let op = f32::from_bits(value_stack.pop::<u32>());
                    value_stack.push::<i32>(op.to_bits() as i32);
                }

                Instr::I64ReinterpretF64 => {
                    let op = f64::from_bits(value_stack.pop::<u64>());
                    value_stack.push::<i64>(op.to_bits() as i64);
                }

                Instr::F32ReinterpretI32 => {
                    let op = value_stack.pop::<i32>();

                    value_stack.push::<f32>(f32::from_bits(op as u32));
                }

                Instr::F64ReinterpretI64 => {
                    let op = value_stack.pop::<i64>();

                    value_stack.push::<f64>(f64::from_bits(op as u64));
                }

                Instr::I32SExtendI8 => {
                    let op = value_stack.pop::<i32>();

                    let op = (op & 0xff) as u8 as i8;
                    value_stack.push::<i32>(op as i32);
                }

                Instr::I32SExtendI16 => {
                    let op = value_stack.pop::<i32>();

                    let op = (op & 0xffff) as u16 as i16;
                    value_stack.push::<i32>(op as i32);
                }

                Instr::I64SExtendI8 => {
                    let op = value_stack.pop::<i64>();

                    let op = (op & 0xff) as u8 as i8;
                    value_stack.push::<i64>(op as i64);
                }

                Instr::I64SExtendI16 => {
                    let op = value_stack.pop::<i64>();

                    let op = (op & 0xffff) as u16 as i16;
                    value_stack.push::<i64>(op as i64);
                }

                Instr::I64SExtendI32 => {
                    let op = value_stack.pop::<i64>();

                    let op = (op & 0xffff_ffff) as u32 as i32;
                    value_stack.push::<i64>(op as i64);
                }

                Instr::I32SConvertSatF32 => {
                    let op = f32::from_bits(value_stack.pop::<u32>());

                    let op = if op.is_nan() {
                        0i32
                    } else if op > (i32::MAX as f32) {
                        i32::MAX
                    } else if op < (i32::MIN as f32) {
                        i32::MIN
                    } else {
                        op as i32
                    };
                    value_stack.push::<i32>(op);
                }

                Instr::I32UConvertSatF32 => {
                    let op = f32::from_bits(value_stack.pop::<u32>());

                    let op = if op.is_nan() {
                        0u32
                    } else if op > (u32::MAX as f32) {
                        u32::MAX
                    } else if op < (u32::MIN as f32) {
                        u32::MIN
                    } else {
                        op as u32
                    };
                    value_stack.push::<i32>(op as i32);
                }

                Instr::I32SConvertSatF64 => {
                    let op = f64::from_bits(value_stack.pop::<u64>());

                    let op = if op.is_nan() {
                        0i32
                    } else if op > (i32::MAX as f64) {
                        i32::MAX
                    } else if op < (i32::MIN as f64) {
                        i32::MIN
                    } else {
                        op as i32
                    };
                    value_stack.push::<i32>(op);
                }

                Instr::I32UConvertSatF64 => {
                    let op = f64::from_bits(value_stack.pop::<u64>());

                    let op = if op.is_nan() {
                        0u32
                    } else if op > (u32::MAX as f64) {
                        u32::MAX
                    } else if op < (u32::MIN as f64) {
                        u32::MIN
                    } else {
                        op as u32
                    };
                    value_stack.push::<i32>(op as i32);
                }

                Instr::I64SConvertSatF32 => {
                    let op = f32::from_bits(value_stack.pop::<u32>());

                    let op = if op.is_nan() {
                        0i64
                    } else if op > (i64::MAX as f32) {
                        i64::MAX
                    } else if op < (i64::MIN as f32) {
                        i64::MIN
                    } else {
                        op as i64
                    };
                    value_stack.push::<i64>(op);
                }

                Instr::I64UConvertSatF32 => {
                    let op = f32::from_bits(value_stack.pop::<u32>());

                    let op = if op.is_nan() {
                        0u64
                    } else if op > (u64::MAX as f32) {
                        u64::MAX
                    } else if op < (u64::MIN as f32) {
                        u64::MIN
                    } else {
                        op as u64
                    };
                    value_stack.push::<i64>(op as i64);
                }

                Instr::I64SConvertSatF64 => {
                    let op = f64::from_bits(value_stack.pop::<u64>());

                    let op = if op.is_nan() {
                        0i64
                    } else if op > (i64::MAX as f64) {
                        i64::MAX
                    } else if op < (i64::MIN as f64) {
                        i64::MIN
                    } else {
                        op as i64
                    };
                    value_stack.push::<i64>(op);
                }

                Instr::I64UConvertSatF64 => {
                    let op = f64::from_bits(value_stack.pop::<u64>());

                    let op = if op.is_nan() {
                        0u64
                    } else if op > (u64::MAX as f64) {
                        u64::MAX
                    } else if op < (u64::MIN as f64) {
                        u64::MIN
                    } else {
                        op as u64
                    };
                    value_stack.push::<i64>(op as i64);
                }

                Instr::CallIntrinsic(TypeIdx(type_idx), idx) => {
                    let external_function = resources.external_functions[*idx].clone();
                    let Type(ResultType(params), ResultType(results)) =
                        &self.types[frame.guest_index][*type_idx as usize];

                    let args: Vec<Value> = params
                        .iter()
                        .enumerate()
                        .map(|(idx, _)| locals.get(&value_stack, idx as u32).into())
                        .collect();

                    let mut results = vec![Value::RefNull; results.len()];

                    drop(resource_lock);
                    (external_function.func)(args.as_slice(), &mut results)?;

                    resource_lock = self
                        .resources
                        .try_lock()
                        .map_err(|_| anyhow::anyhow!("failed to lock resources"))?;
                    resources = resource_lock.deref_mut();

                    for result in results {
                        value_stack.push_value(result.into());
                    }
                }
            }
            frame.pc += 1;
        }

        let mut output = Vec::with_capacity(results.len());
        for vt in results.iter().rev() {
            let next = value_stack.pop_valtype(*vt).into();
            output.push(next);
        }
        output.reverse();
        Ok(output)
    }

    fn compute_constant_expr(
        &self,
        module_idx: GuestIndex,
        instrs: &[Instr],
        resources: &mut Resources,
    ) -> anyhow::Result<StackValue> {
        Ok(match instrs.first() {
            Some(Instr::F32Const(c)) => StackValue::F32(*c),
            Some(Instr::F64Const(c)) => StackValue::F64(*c),
            Some(Instr::I32Const(c)) => StackValue::I32(*c),
            Some(Instr::I64Const(c)) => StackValue::I64(*c),
            Some(Instr::GlobalGet(c)) => {
                let globalidx = self.global(module_idx, *c);
                resources
                    .global_values
                    .get(globalidx)
                    .ok_or_else(|| anyhow::anyhow!("uninitialized global"))?
                    .clone()
            }

            Some(Instr::RefNull(RefType::FuncRef)) => StackValue::RefFunc(None),
            Some(Instr::RefNull(RefType::ExternRef)) => StackValue::RefExtern(None),
            Some(Instr::RefFunc(FuncIdx(c))) => StackValue::RefFunc(NonZeroU64::new(
                ((module_idx as u64 + 1) & 0xffff_ffff) << 32 | (*c as u64),
            )),
            _ => anyhow::bail!("unsupported instruction"),
        })
    }
}

static CANON_32BIT_NAN: u32 = 0b01111111110000000000000000000000;
static CANON_64BIT_NAN: u64 = 0b0111111111111000000000000000000000000000000000000000000000000000;
// Portions copied from wasmtime: https://github.com/bytecodealliance/wasmtime/blob/main/crates/wasmtime/src/runtime/vm/libcalls.rs#L709
const TOINT_32: f32 = 1.0 / f32::EPSILON;
const TOINT_64: f64 = 1.0 / f64::EPSILON;

// NB: replace with `round_ties_even` from libstd when it's stable as
// tracked by rust-lang/rust#96710
pub extern "C" fn nearestf32(x: f32) -> f32 {
    // Rust doesn't have a nearest function; there's nearbyint, but it's not
    // stabilized, so do it manually.
    // Nearest is either ceil or floor depending on which is nearest or even.
    // This approach exploited round half to even default mode.
    let i = x.to_bits();
    let e = i >> 23 & 0xff;
    if e >= 0x7f_u32 + 23 {
        // Check for NaNs.
        if e == 0xff {
            // Read the 23-bits significand.
            if i & 0x7fffff != 0 {
                // Ensure it's arithmetic by setting the significand's most
                // significant bit to 1; it also works for canonical NaNs.
                return f32::from_bits(i | (1 << 22));
            }
        }
        x
    } else {
        f32::copysign(f32::abs(x) + TOINT_32 - TOINT_32, x)
    }
}

pub extern "C" fn nearestf64(x: f64) -> f64 {
    let i = x.to_bits();
    let e = i >> 52 & 0x7ff;
    if e >= 0x3ff_u64 + 52 {
        // Check for NaNs.
        if e == 0x7ff {
            // Read the 52-bits significand.
            if i & 0xfffffffffffff != 0 {
                // Ensure it's arithmetic by setting the significand's most
                // significant bit to 1; it also works for canonical NaNs.
                return f64::from_bits(i | (1 << 51));
            }
        }
        x
    } else {
        f64::copysign(f64::abs(x) + TOINT_64 - TOINT_64, x)
    }
}
