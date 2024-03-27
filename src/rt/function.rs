use crate::nodes::{CodeIdx, TypeIdx, Import};

use super::{TKTK, imports::Imports};

#[derive(Debug)]
pub(crate) enum FuncInstImpl {
    Guest(CodeIdx),
    Host(TKTK),
}

#[derive(Debug)]
pub(crate) struct FuncInst {
    r#type: TypeIdx,
    pub(crate) r#impl: FuncInstImpl,
}

impl FuncInst {
    pub(crate) fn resolve(ty: TypeIdx, import: &Import<'_>, imports: &Imports) -> anyhow::Result<Self> {
        // TODO: actually resolve!
        Ok(Self {
            r#type: ty,
            r#impl: FuncInstImpl::Host(TKTK)
        })
    }

    pub(crate) fn new(ty: TypeIdx, code_idx: CodeIdx) -> Self {
        Self {
            r#type: ty,
            r#impl: FuncInstImpl::Guest(code_idx)
        }
    }

    #[inline]
    pub(crate) fn typeidx(&self) -> &TypeIdx {
        &self.r#type
    }
}

