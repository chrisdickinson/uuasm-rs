use std::collections::HashMap;

use crate::{
    intern_map::InternMap,
    nodes::{ExportDesc, FuncIdx, GlobalIdx, Import, MemIdx, Module, TableIdx},
    rt::machine::Machine,
};

type HostFnIndex = usize;
pub(super) type GuestIndex = usize;

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub(crate) struct ExternKey(usize, usize);

#[derive(Debug, Clone, Copy)]
pub(crate) enum Extern {
    Func(GuestIndex, FuncIdx),
    Global(GuestIndex, GlobalIdx),
    Table(GuestIndex, TableIdx),
    Memory(GuestIndex, MemIdx),
}

#[derive(Debug, Default, Clone)]
pub(crate) struct Imports<'a> {
    // borrow a page from wasmtime!
    pub(crate) guests: Vec<Module<'a>>,
    pub(crate) externs: HashMap<ExternKey, Extern>,
    pub(super) internmap: InternMap,
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

    pub(crate) fn link_module(&mut self, modname: &str, module: Module<'a>) {
        let idx = self.guests.len();
        for export in module.export_section().unwrap_or_default() {
            match export.desc {
                ExportDesc::Func(func_idx) => {
                    self.link_extern(modname, export.nm.0, Extern::Func(idx, func_idx));
                }
                ExportDesc::Table(table_idx) => {
                    self.link_extern(modname, export.nm.0, Extern::Table(idx, table_idx));
                }
                ExportDesc::Mem(mem_idx) => {
                    self.link_extern(modname, export.nm.0, Extern::Memory(idx, mem_idx));
                }
                ExportDesc::Global(global_idx) => {
                    self.link_extern(modname, export.nm.0, Extern::Global(idx, global_idx));
                }
            }
        }
        self.guests.push(module);
    }

    pub(crate) fn instantiate(mut self, module: Module<'a>) -> anyhow::Result<Machine<'a>> {
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
    use std::mem::size_of;

    use crate::{parse::parse, rt::Value};

    use super::*;

    #[test]
    fn imports_create() -> anyhow::Result<()> {
        let imports = Imports::new();
        let wasm = parse(include_bytes!("../../example.wasm"))?;

        let mut machine = imports.instantiate(wasm)?;

        let exports: Vec<_> = machine.exports().collect();
        eprintln!("{exports:?}");
        let result = machine.call("add_i32", &[Value::I32(5), Value::I32(2)])?;
        eprintln!("{result:?}");

        eprintln!("instr={}", size_of::<Machine>());
        Ok(())
    }
}
