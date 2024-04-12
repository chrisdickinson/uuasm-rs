use crate::nodes::{TableType, Import, FuncIdx};

use super::{value::Value, imports::{Imports, ExternTable, Extern}, machine::MachineTableIndex};

#[derive(Debug, Clone)]
enum TableInstImpl {
    Local(MachineTableIndex),
    Remote(ExternTable),
}

#[derive(Debug, Clone)]
pub(crate) struct TableInst {
    r#type: TableType,
    r#impl: TableInstImpl,
}

impl TableInst {
    pub(crate) fn resolve(ty: TableType, import: &Import<'_>, imports: &Imports) -> anyhow::Result<Self> {
        let Some(ext) = imports.lookup(import) else {
            anyhow::bail!("could not resolve {}/{}", import.r#mod.0, import.nm.0);
        };

        let Extern::Table(table) = ext else {
            anyhow::bail!("expected {}/{} to resolve to a memory", import.r#mod.0, import.nm.0);
        };

        // TODO: validate type.
        Ok(Self {
            r#type: ty,
            r#impl: TableInstImpl::Remote(table)
        })
    }

    pub(crate) fn new(ty: TableType, idx: MachineTableIndex) -> Self {
        Self {
            r#type: ty,
            r#impl: TableInstImpl::Local(idx),
        }
    }

    /*
    pub(crate) fn get(&self, idx: usize) -> Option<Value> {
        match &self.r#impl {
            TableInstImpl::Local(values) => {
                values.get(idx).copied()
            }
            TableInstImpl::Remote(_) => todo!(),
        }
    }

    pub(crate) fn write_func_indices(&mut self, func_indices: &[FuncIdx]) {
        match &mut self.r#impl {
            TableInstImpl::Local(v) => {
                for (idx, xs) in func_indices.iter().enumerate() {
                    v[idx] = Value::RefFunc(*xs);
                }
            }
            TableInstImpl::Remote(_) => {
                todo!("TODO: handle populating imported tables with elements")
            }
        }
    }
    */
}

