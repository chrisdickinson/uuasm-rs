use crate::nodes::{GlobalType, Import, Mutability};

use super::{value::Value, imports::{Imports, ExternGlobal}};

#[derive(Debug, Clone)]
enum GlobalInstImpl {
    Local(Value),
    Remote(ExternGlobal),
}

#[derive(Debug, Clone)]
pub(crate) struct GlobalInst {
    r#type: GlobalType,
    r#impl: GlobalInstImpl,
}

impl GlobalInst {
    #[inline]
    pub(crate) fn value(&self) -> Value {
        match self.r#impl {
            GlobalInstImpl::Local(v) => v,
            GlobalInstImpl::Remote(_) => todo!(),
        }
    }

    #[inline]
    pub(crate) fn value_mut(&mut self) -> &mut Value {
        match &mut self.r#impl {
            GlobalInstImpl::Local(v) => v,
            GlobalInstImpl::Remote(_) => todo!(),
        }
    }

    pub(crate) fn assign(&mut self, v: &Value) -> anyhow::Result<()> {
        if self.r#type.1 == Mutability::Const {
            anyhow::bail!("cannot assign to constant global");
        }

        self.r#type.0.validate(&v)?;
        *self.value_mut() = *v;
        Ok(())
    }

    pub(crate) fn resolve(ty: GlobalType, import: &Import<'_>, imports: &Imports) -> anyhow::Result<Self> {
        todo!()
    }

    pub(crate) fn new(ty: GlobalType, v: Value) -> anyhow::Result<Self> {
        ty.0.validate(&v)?;
        Ok(Self {
            r#type: ty,
            r#impl: GlobalInstImpl::Local(v)
        })
    }
}

