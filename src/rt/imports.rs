use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{
    intern_map::InternMap,
    nodes::{
        Code, Export, ExportDesc, Expr, Func, FuncIdx, GlobalIdx, Import, Instr, MemIdx, Module,
        ModuleBuilder, Name, ResultType, TableIdx, Type, TypeIdx,
    },
    rt::machine::Machine,
};

use super::{machine::ExternalFunction, Value};

type HostFnIndex = usize;
pub(super) type GuestIndex = usize;

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub(crate) struct ExternKey(pub(crate) usize, pub(crate) usize);

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
    pub(super) modname_to_guest_idx: HashMap<usize, HashSet<usize>>,
    pub(crate) external_functions: Vec<ExternalFunction>,
}

pub(crate) trait LookupImport {
    fn lookup(&self, import: &Import) -> Option<Extern>;
}

impl<'a> LookupImport for Imports<'a> {
    fn lookup(&self, import: &Import) -> Option<Extern> {
        let modname = self.internmap.get(import.r#mod.0)?;
        let name = self.internmap.get(import.nm.0)?;
        let key = ExternKey(modname, name);
        self.externs.get(&key).copied()
    }
}

impl<'a> Imports<'a> {
    pub(crate) fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    fn link_extern(&mut self, modname: &str, name: &str, ext: Extern) {
        let modname = self.internmap.insert(modname);
        let name = self.internmap.insert(name);

        self.externs.insert(ExternKey(modname, name), ext);
    }

    pub(crate) fn link_module(&mut self, modname: &str, module: Module<'a>) {
        let idx = self.guests.len();
        let modname_idx = self.internmap.insert(modname);
        self.modname_to_guest_idx
            .entry(modname_idx)
            .or_insert_with(HashSet::new)
            .insert(idx);

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

    pub(crate) fn link_hostfn(
        &mut self,
        modname: &str,
        funcname: &str,
        typedef: Type,
        func: impl Fn(&[Value], &mut [Value]) -> anyhow::Result<()> + Send + Sync + 'static,
    ) {
        let modname = self.internmap.insert(modname);
        let name = self.internmap.insert(funcname);

        let idx = self.guests.len();

        let module = ModuleBuilder::new()
            .type_section(vec![typedef.clone()])
            .function_section(vec![TypeIdx(0)])
            .code_section(vec![Code(Func {
                locals: typedef.clone().into(),
                expr: Expr(vec![
                    Instr::CallIntrinsic(self.external_functions.len()),
                    Instr::Return,
                ]),
            })])
            .build();

        self.externs
            .insert(ExternKey(modname, name), Extern::Func(idx, FuncIdx(0)));
        self.guests.push(module);

        self.external_functions.push(ExternalFunction {
            func: Arc::new(func),
            typedef,
        });
    }

    pub(crate) fn instantiate(self) -> anyhow::Result<Machine<'a>> {
        Machine::new(
            self.guests,
            self.externs,
            self.internmap,
            self.external_functions,
            self.modname_to_guest_idx,
        )
    }
}

#[cfg(test)]
mod test {
    use std::mem::size_of;

    use crate::{
        nodes::{NumType, ValType},
        parse::parse,
        rt::Value,
    };

    use super::*;

    #[test]
    fn imports_create() -> anyhow::Result<()> {
        let mut imports = Imports::new();
        let wasm = parse(include_bytes!("../../example.wasm"))?;

        imports.link_module("env", wasm);
        let mut machine = imports.instantiate()?;

        // let exports: Vec<_> = machine.exports().collect();
        // eprintln!("{exports:?}");
        let result = machine.call("env", "add_i32", &[Value::I32(5), Value::I32(2)])?;
        eprintln!("{result:?}");

        eprintln!("instr={}", size_of::<Machine>());
        Ok(())
    }

    #[test]
    fn host_imports() -> anyhow::Result<()> {
        let mut imports = Imports::new();
        let wasm = parse(include_bytes!("../../example3.wasm"))?;

        imports.link_hostfn(
            "env",
            "add",
            Type(
                ResultType(
                    vec![
                        ValType::NumType(NumType::I32),
                        ValType::NumType(NumType::I32),
                    ]
                    .into(),
                ),
                ResultType(vec![ValType::NumType(NumType::I32)].into()),
            ),
            |stack: &[Value], results: &mut [Value]| {
                eprintln!("stack={stack:?}");
                let Value::I32(lhs) = stack[0] else {
                    anyhow::bail!("expected i32 value");
                };
                let Value::I32(rhs) = stack[1] else {
                    anyhow::bail!("expected i32 value");
                };
                results[0] = Value::I32(lhs + rhs);
                Ok(())
            },
        );
        imports.link_module("env", wasm);
        let mut machine = imports.instantiate()?;

        // let exports: Vec<_> = machine.exports().collect();
        // eprintln!("{exports:?}");
        let result = machine.call("env", "foo", &[])?;
        eprintln!("{result:?}");

        eprintln!("instr={}", size_of::<Machine>());
        Ok(())
    }
}
