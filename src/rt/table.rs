use crate::nodes::{TableType, Import, FuncIdx};

use super::{value::Value, TKTK, imports::Imports};

#[derive(Debug)]
enum TableInstImpl {
    Local(Vec<Value>),
    Remote(TKTK),
}

#[derive(Debug, Clone)]
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
            r#impl: TableInstImpl::Local(vec![
                Value::RefNull;
                ty.1.min() as usize
            ]),
        }
    }
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
}

