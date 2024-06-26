use anyhow::Context;
use std::collections::HashMap;

use crate::nodes::{
    Code, CodeIdx, Data, Elem, Export, Expr, FuncIdx, Global, GlobalIdx, ImportDesc, Instr,
    Module as ParsedModule, Type, TypeIdx,
};

use super::{
    function::FuncInst,
    global::GlobalInst,
    imports::{GuestIndex, Imports},
    instance::ModuleInstance,
    machine::MachineBuilder,
    memory::MemInst,
    table::TableInst,
    value::Value,
    TKTK,
};

#[derive(Debug, Clone)]
pub(crate) struct Module {
    parsed_module: ParsedModule,
}

impl Module {
    pub(crate) fn new<'b: 'a>(module: ParsedModule) -> Self {
        Self {
            parsed_module: module,
        }
    }

    pub(crate) fn exports(&self) -> &[Export] {
        self.parsed_module.export_section().unwrap_or_default()
    }

    pub(crate) fn typedef(&self, idx: &TypeIdx) -> Option<&Type> {
        self.parsed_module
            .type_section()
            .and_then(|types| types.get(idx.0 as usize))
    }

    pub(crate) fn instrs(&self, idx: CodeIdx) -> Option<&Code> {
        self.parsed_module
            .code_section()
            .iter()
            .flat_map(|xs| xs.iter())
            .nth(idx.0 as usize)
    }

    pub(crate) fn resolve(&self, builder: &mut MachineBuilder) -> anyhow::Result<ModuleInstance> {
        for imp in self
            .parsed_module
            .import_section()
            .iter()
            .flat_map(|xs| xs.iter())
        {
            match imp.desc {
                ImportDesc::Func(desc) => builder.define_function_import(desc, imp),
                ImportDesc::Mem(desc) => builder.define_memory_import(desc, imp),
                ImportDesc::Table(desc) => builder.define_table_import(desc, imp),
                ImportDesc::Global(desc) => builder.define_global_import(desc, imp),
            }
        }

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
                    Some(Instr::GlobalGet(GlobalIdx(idx))) => {
                        let idx = *idx as usize;
                        globals[idx].value()
                    }
                    Some(Instr::RefNull(_ref_type)) => Value::RefNull,
                    Some(Instr::RefFunc(FuncIdx(idx))) => Value::RefFunc(FuncIdx(*idx)),
                    _ => anyhow::bail!("unsupported global initializer instruction"),
                };

                global_type
                    .0
                    .validate(&global)
                    .context("global type does not accept this value")?;
                globals.push(GlobalInst::new(*global_type, global)?);
                Ok(globals)
            })?;

        // next step: build our function table.
        let functions = func_imports
            .into_iter()
            .chain(
                self.parsed_module
                    .function_section()
                    .iter()
                    .flat_map(|xs| xs.iter())
                    .copied()
                    .enumerate()
                    .map(|(code_idx, xs)| {
                        if code_idx
                            >= self
                                .parsed_module
                                .code_section()
                                .map(|xs| xs.len())
                                .unwrap_or_default()
                        {
                            anyhow::bail!("code idx out of range");
                        }

                        Ok(FuncInst::new(xs, CodeIdx(code_idx as u32)))
                    }),
            )
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut memories: Vec<_> = memory_imports
            .into_iter()
            .chain(
                self.parsed_module
                    .memory_section()
                    .iter()
                    .flat_map(|xs| xs.iter())
                    .map(|memtype| Ok(MemInst::new(imports.register_memory(*memtype)))),
            )
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut tables: Vec<_> = table_imports
            .into_iter()
            .chain(
                self.parsed_module
                    .table_section()
                    .iter()
                    .flat_map(|xs| xs.iter())
                    .map(|tabletype| Ok(TableInst::new(imports.register_table(*tabletype)))),
            )
            .collect::<anyhow::Result<Vec<_>>>()?;

        // TODO:
        // - [x] apply data to memories
        // - [ ] apply elems to tables
        for data in self
            .parsed_module
            .data_section()
            .iter()
            .flat_map(|xs| xs.iter())
        {
            match data {
                Data::Active(data, memory_idx, expr) => {
                    let offset = compute_constant_expr(expr, globals.as_slice())?;
                    let offset = offset
                        .as_usize()
                        .ok_or_else(|| anyhow::anyhow!("expected i32 or i64"))?;

                    memories[memory_idx.0 as usize].copy_data(data, offset)?;
                }
                Data::Passive(_) => continue,
            }
        }

        for elem in self
            .parsed_module
            .element_section()
            .iter()
            .flat_map(|xs| xs.iter())
        {
            match elem {
                Elem::ActiveSegmentFuncs(expr, func_indices) => {
                    let offset = compute_constant_expr(expr, globals.as_slice())?;
                    let offset = offset
                        .as_usize()
                        .ok_or_else(|| anyhow::anyhow!("expected i32 or i64"))?;

                    let Some(table) = tables.get_mut(offset) else {
                        anyhow::bail!("could not populate elements: no table at idx={}", offset);
                    };

                    table.write_func_indices(func_indices.as_slice());
                }
                Elem::PassiveSegment(_, _) => todo!("PassiveSegment"),
                Elem::ActiveSegment(_, _, _, _) => todo!("ActiveSegment"),
                Elem::DeclarativeSegment(_, _) => todo!("DeclarativeSegment"),
                Elem::ActiveSegmentExpr(_, _) => todo!("ActiveSegmentExpr"),
                Elem::PassiveSegmentExpr(_, _) => todo!("PassiveSegmentExpr"),
                Elem::ActiveSegmentTableAndExpr(_, _, _, _) => todo!("ActiveSegmentTableAndExpr"),
                Elem::DeclarativeSegmentExpr(_, _) => todo!("DeclarativeSegmentExpr"),
            }
        }

        Ok(ModuleInstance {
            module: module_idx,
            functions,
            globals,
            memories,
            tables,
        })
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
            global.value()
        }

        Some(Instr::RefNull(_c)) => todo!(),
        Some(Instr::RefFunc(c)) => Value::RefFunc(*c),
        _ => anyhow::bail!("unsupported instruction"),
    })
}
