use std::{sync::Arc, collections::HashMap};

use crate::{nodes::{ExportDesc, FuncIdx, Import, TableIdx, MemIdx, GlobalIdx}, rt::machine::Machine};

use super::{value::Value, module::Module};

type HostFnIndex = usize;
type HostTableIndex = usize;
type HostGlobalIndex = usize;
type HostMemoryIndex = usize;
pub(super) type GuestIndex = usize;

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub(crate) struct ExternKey(usize, usize);

#[derive(Debug, Clone, Copy)]
pub(crate) enum ExternGlobal {
    Guest(GuestIndex, GlobalIdx),
    Host(HostGlobalIndex), 
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ExternFunc {
    Guest(GuestIndex, FuncIdx),
    Host(HostFnIndex), 
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ExternTable {
    Guest(GuestIndex, TableIdx),
    Host(HostTableIndex), 
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ExternMemory {
    Guest(GuestIndex, MemIdx),
    Host(HostMemoryIndex), 
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Extern {
    Func(ExternFunc),
    Global(ExternGlobal),
    Table(ExternTable),
    Memory(ExternMemory),
}

#[derive(Debug, Default, Clone)]
pub(crate) struct Imports<'a> {
    // borrow a page from wasmtime!
    pub(crate) strings: Vec<Arc<str>>,
    string_to_idx: HashMap<Arc<str>, usize>,
    pub(crate) guests: Vec<Module<'a>>,
    pub(crate) externs: HashMap<ExternKey, Extern>,
}

impl<'a> Imports<'a> {
    pub(crate) fn new(_globals: Vec<Value>) -> Self {
        Self {
            ..Default::default()
        }
    }

    pub(crate) fn lookup(&self, import: &Import) -> Option<Extern> {
        let modname = self.string_to_idx.get(import.r#mod.0)?;
        let name = self.string_to_idx.get(import.nm.0)?;
        let key = ExternKey(*modname, *name);
        self.externs.get(&key).copied()
    }

    fn link_extern(&mut self, modname: &str, name: &str, ext: Extern) {
        let modname = self.intern(modname);
        let name = self.intern(name);

        self.externs.insert(ExternKey(modname, name), ext);
    }

    fn link_module(&mut self, modname: &str, module: Module<'a>) {
        let idx = self.guests.len();
        for export in module.exports() {
            match export.desc {
                ExportDesc::Func(func_idx) => {
                    self.link_extern(modname, export.nm.0, Extern::Func(ExternFunc::Guest(idx, func_idx)));
                },
                ExportDesc::Table(table_idx) => {
                    self.link_extern(modname, export.nm.0, Extern::Table(ExternTable::Guest(idx, table_idx)));
                },
                ExportDesc::Mem(mem_idx) => {
                    self.link_extern(modname, export.nm.0, Extern::Memory(ExternMemory::Guest(idx, mem_idx)));
                },
                ExportDesc::Global(global_idx) => {
                    self.link_extern(modname, export.nm.0, Extern::Global(ExternGlobal::Guest(idx, global_idx)));
                },
            }
        }
        self.guests.push(module);
    }

    fn instantiate(mut self, module: Module<'a>) -> anyhow::Result<Machine<'a>> {
        let mut guests = self.guests.split_off(0);
        guests.push(module);

        // Reserve guest tables, memories, globals, and funcs...
        // what are we doing at this point? the imports are about to be subsumed by the machine.
        // should we be passing the machine around instead??
        // ... well, maybe. hm.
        let instances = guests.iter().enumerate().map(|(idx, module)| module.instantiate(&mut self, idx)).collect::<anyhow::Result<Vec<_>>>()?;

        // OK, machines have to collect a list of modules AND a list of instances
        // AND ModuleInstances cannot refer to Modules directly, but rather indirectly through
        // indices into the machine
        Ok(Machine::new(self, instances))
    }

    fn intern(&mut self, string: &str) -> usize {
        if let Some(idx) = self.string_to_idx.get(string) {
            return *idx
        }

        let string: Arc<str> = string.into();
        let idx = self.strings.len();
        self.strings.push(string.clone());
        self.string_to_idx.insert(string, idx);
        idx
    }
}
