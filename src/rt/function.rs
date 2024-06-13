use crate::nodes::{CodeIdx, FuncIdx, Import, TypeIdx};

use super::imports::{Extern, GuestIndex, LookupImport};

#[derive(Debug, Clone)]
pub(crate) enum FuncInstImpl {
    Local(CodeIdx),
    Remote(GuestIndex, FuncIdx),
}

#[derive(Debug, Clone)]
pub(crate) struct FuncInst {
    r#type: TypeIdx,
    pub(crate) r#impl: FuncInstImpl,
}

impl FuncInst {
    pub(crate) fn resolve(
        ty: TypeIdx,
        import: &Import<'_>,
        imports: &impl LookupImport,
    ) -> anyhow::Result<Self> {
        let Some(ext) = imports.lookup(import) else {
            anyhow::bail!("could not resolve {}/{}", import.r#mod.0, import.nm.0);
        };

        let Extern::Func(guest_idx, func_idx) = ext else {
            anyhow::bail!(
                "expected {}/{} to resolve to a func",
                import.r#mod.0,
                import.nm.0
            );
        };

        // TODO: validate type.
        Ok(Self {
            r#type: ty,
            r#impl: FuncInstImpl::Remote(guest_idx, func_idx),
        })
    }

    pub(crate) fn new(ty: TypeIdx, code_idx: CodeIdx) -> Self {
        Self {
            r#type: ty,
            r#impl: FuncInstImpl::Local(code_idx),
        }
    }

    #[inline]
    pub(crate) fn codeidx(&self) -> CodeIdx {
        match self.r#impl {
            FuncInstImpl::Local(code_idx) => code_idx,
            FuncInstImpl::Remote(_, _) => unreachable!(),
        }
    }

    #[inline]
    pub(crate) fn typeidx(&self) -> &TypeIdx {
        &self.r#type
    }
}
