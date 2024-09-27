use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use uuasm_ir::{
    Code, ExportDesc, Expr, Func, FuncIdx, GlobalIdx, Import, Instr, MemIdx, Module, ModuleBuilder,
    TableIdx, Type, TypeIdx,
};

use crate::{intern_map::InternMap, machine::Machine};

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
pub struct Imports {
    // borrow a page from wasmtime!
    pub(crate) guests: Vec<Module>,
    pub(crate) externs: HashMap<ExternKey, Extern>,
    pub(super) internmap: InternMap,
    pub(super) modname_to_guest_idx: HashMap<usize, HashSet<usize>>,
    pub(crate) external_functions: Vec<ExternalFunction>,
}

pub(crate) trait LookupImport {
    fn lookup(&self, import: &Import) -> Option<Extern>;
}

impl LookupImport for Imports {
    fn lookup(&self, import: &Import) -> Option<Extern> {
        let modname = self.internmap.get(import.module())?;
        let name = self.internmap.get(import.name())?;
        let key = ExternKey(modname, name);
        self.externs.get(&key).copied()
    }
}

impl Imports {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    fn link_extern(&mut self, modname: &str, name: &str, ext: Extern) {
        let modname = self.internmap.insert(modname);
        let name = self.internmap.insert(name);

        self.externs.insert(ExternKey(modname, name), ext);
    }

    pub fn link_module(&mut self, modname: &str, module: Module) {
        let idx = self.guests.len();
        let modname_idx = self.internmap.insert(modname);
        self.modname_to_guest_idx
            .entry(modname_idx)
            .or_default()
            .insert(idx);

        for export in module.export_section().unwrap_or_default() {
            match export.desc() {
                ExportDesc::Func(func_idx) => {
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
        self.guests.push(module);
    }

    pub fn link_hostfn(
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
            .type_section(vec![typedef.clone()].into())
            .function_section(vec![TypeIdx(0)].into())
            .code_section(
                vec![Code(Func {
                    locals: vec![].into(),
                    expr: Expr(vec![
                        Instr::CallIntrinsic(TypeIdx(0), self.external_functions.len()),
                        Instr::Return,
                    ]),
                })]
                .into(),
            )
            .build();

        self.externs
            .insert(ExternKey(modname, name), Extern::Func(idx, FuncIdx(0)));
        self.guests.push(module);

        self.external_functions.push(ExternalFunction {
            func: Arc::new(func),
            typedef,
        });
    }

    pub fn instantiate(self) -> anyhow::Result<Machine> {
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

    use crate::Value;
    use uuasm_codec::parse;
    use uuasm_ir::{DefaultIRGenerator, NumType, ResultType, ValType};

    use super::*;

    #[test]
    fn imports_create() -> anyhow::Result<()> {
        let mut imports = Imports::new();
        let wasm = parse(
            DefaultIRGenerator::new(),
            include_bytes!("../../../corpus/example.wasm"),
        )?;

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
        let wasm = parse(
            DefaultIRGenerator::new(),
            include_bytes!("../../../corpus/example3.wasm"),
        )?;

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
