use crate::nodes::{Import, TableIdx, TableType};

use super::{
    imports::{Extern, GuestIndex, Imports, LookupImport},
    machine::MachineTableIndex,
};

#[derive(Debug, Clone)]
pub(super) enum TableInstImpl {
    Local(MachineTableIndex),
    Remote(GuestIndex, TableIdx),
}

#[derive(Debug, Clone)]
pub(crate) struct TableInst {
    r#type: TableType,
    pub(super) r#impl: TableInstImpl,
}

impl TableInst {
    pub(crate) fn resolve(
        ty: TableType,
        import: &Import,
        imports: &impl LookupImport,
    ) -> anyhow::Result<Self> {
        let Some(ext) = imports.lookup(import) else {
            anyhow::bail!("could not resolve {}/{}", import.r#mod.0, import.nm.0);
        };

        let Extern::Table(module_idx, table_idx) = ext else {
            anyhow::bail!(
                "expected {}/{} to resolve to a memory",
                import.r#mod.0,
                import.nm.0
            );
        };

        // TODO: validate type.
        Ok(Self {
            r#type: ty,
            r#impl: TableInstImpl::Remote(module_idx, table_idx),
        })
    }

    pub(crate) fn new(ty: TableType, idx: MachineTableIndex) -> Self {
        Self {
            r#type: ty,
            r#impl: TableInstImpl::Local(idx),
        }
    }

    pub(crate) fn typedef(&self) -> &TableType {
        &self.r#type
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
