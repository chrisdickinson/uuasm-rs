use std::{sync::Arc, collections::HashMap};

use crate::{nodes::{ExportDesc, Module, FuncIdx, Import, TableIdx, MemIdx, GlobalIdx}, rt::machine::Machine};

use super::machine::{MachineGlobalIndex, MachineTableIndex, MachineMemoryIndex};

type HostFnIndex = usize;
pub(super) type GuestIndex = usize;

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub(crate) struct ExternKey(usize, usize);


// So here's an interesting thing... there's no need to keep a MachineGlobalIndex etc if we
// _generate_ "host" modules to hold our host memories, tables, globals, and functions. the
// functions should just point to single-instr "CallTrampoline" impls.

#[derive(Debug, Clone, Copy)]
pub(crate) enum ExternGlobal {
    Guest(GuestIndex, GlobalIdx),
    Host(MachineGlobalIndex), 
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ExternFunc {
    Guest(GuestIndex, FuncIdx),
    Host(HostFnIndex),
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ExternTable {
    Guest(GuestIndex, TableIdx),
    Host(MachineTableIndex), 
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ExternMemory {
    Guest(GuestIndex, MemIdx),
    Host(MachineMemoryIndex), 
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Extern {
    Func(ExternFunc),
    Global(ExternGlobal),
    Table(ExternTable),
    Memory(ExternMemory),
}

#[derive(Default, Debug, Clone)]
pub(super) struct InternMap {
    strings: Vec<Arc<str>>,
    string_to_idx: HashMap<Arc<str>, usize>,
}

impl InternMap {
    pub(super) fn get(&self, key: &str) -> Option<usize> {
        self.string_to_idx.get(key).copied()
    }

    pub(super) fn idx(&self, key: usize) -> Option<&str> {
        self.strings.get(key).map(|xs| xs.as_ref())
    }

    pub(super) fn insert(&mut self, string: &str) -> usize {
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

#[derive(Debug, Default, Clone)]
pub(crate) struct Imports<'a> {
    // borrow a page from wasmtime!
    pub(crate) guests: Vec<Module<'a>>,
    pub(crate) externs: HashMap<ExternKey, Extern>,
    pub(super) internmap: InternMap
}

impl<'a> Imports<'a> {
    pub(crate) fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    pub(crate) fn lookup(&self, import: &Import) -> Option<Extern> {
        let modname = self.internmap.get(import.r#mod.0)?;
        let name = self.internmap.get(import.nm.0)?;
        let key = ExternKey(modname, name);
        self.externs.get(&key).copied()
    }

    fn link_extern(&mut self, modname: &str, name: &str, ext: Extern) {
        let modname = self.internmap.insert(modname);
        let name = self.internmap.insert(name);

        self.externs.insert(ExternKey(modname, name), ext);
    }

    fn link_module(&mut self, modname: &str, module: Module<'a>) {
        let idx = self.guests.len();
        for export in module.export_section().unwrap_or_default() {
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
        let mut exportmap = HashMap::new();
        for export in module.export_section().unwrap_or_default() {
            exportmap.insert(self.internmap.insert(export.nm.0), export.desc);
        }

        self.guests.push(module);

        Machine::new(self, exportmap)
    }
}

#[cfg(test)]
mod test {
    use crate::{parse::parse, rt::Value};

    use super::*;

    #[test]
    fn imports_create() -> anyhow::Result<()> {
        let imports = Imports::new();
        let wasm = parse(include_bytes!("../../example.wasm"))?;

        let mut machine = imports.instantiate(wasm)?;

        let exports: Vec<_> = machine.exports().collect();
        eprintln!("{exports:?}");
        machine.call("foo", &[Value::I32(0)])?;

        Ok(())
    }
}
