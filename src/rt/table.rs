use crate::nodes::{TableType, Import, FuncIdx};

use super::{value::Value, TKTK, imports::Imports};

#[derive(Debug)]
enum TableInstImpl {
    Guest(Vec<Value>),
    Host(TKTK),
}

#[derive(Debug)]
pub(crate) struct TableInst {
    r#type: TableType,
    r#impl: TableInstImpl,
}

impl TableInst {
    pub(crate) fn resolve(ty: TableType, import: &Import<'_>, imports: &Imports) -> anyhow::Result<Self> {
        todo!()
    }

    pub(crate) fn new(ty: TableType) -> Self {
        Self {
            r#type: ty,
            r#impl: TableInstImpl::Guest(vec![
                Value::RefNull;
                ty.1.min() as usize
            ]),
        }
    }
    pub(crate) fn get(&self, idx: usize) -> Option<Value> {
        match &self.r#impl {
            TableInstImpl::Guest(values) => {
                values.get(idx).copied()
            }
            TableInstImpl::Host(_) => todo!(),
        }
    }

    pub(crate) fn write_func_indices(&mut self, func_indices: &[FuncIdx]) {
        match &mut self.r#impl {
            TableInstImpl::Guest(v) => {
                for (idx, xs) in func_indices.iter().enumerate() {
                    v[idx] = Value::RefFunc(*xs);
                }
            }
            TableInstImpl::Host(_) => {
                todo!("TODO: handle populating imported tables with elements")
            }
        }
    }
}

