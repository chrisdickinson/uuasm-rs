use crate::nodes::{CodeIdx, TypeIdx, Import, Func};

use super::{imports::{Imports, Extern, ExternFunc}, machine::{MachineCodeIndex, Machine}};

#[derive(Debug, Clone)]
pub(crate) enum FuncInstImpl {
    Local(CodeIdx),
    Remote(ExternFunc),
}

#[derive(Debug, Clone)]
pub(crate) struct FuncInst {
    r#type: TypeIdx,
    pub(crate) r#impl: FuncInstImpl,
}

impl FuncInst {
    pub(crate) fn resolve(ty: TypeIdx, import: &Import<'_>, imports: &Imports) -> anyhow::Result<Self> {
        let Some(ext) = imports.lookup(import) else {
            anyhow::bail!("could not resolve {}/{}", import.r#mod.0, import.nm.0);
        };

        let Extern::Func(func) = ext else {
            anyhow::bail!("expected {}/{} to resolve to a func", import.r#mod.0, import.nm.0);
        };

        // TODO: validate type.
        Ok(Self {
            r#type: ty,
            r#impl: FuncInstImpl::Remote(func)
        })
    }

    pub(crate) fn func<'a>(&self, machine: &Machine<'a>, module_idx: usize) -> Option<&Func> {
        match self.r#impl {
            FuncInstImpl::Local(code_idx) => machine.code(module_idx, code_idx.0 as usize).map(|xs| &xs.0),
            FuncInstImpl::Remote(external) => {
                match external {
                    ExternFunc::Guest(module_idx, offset) => {
                        machine.function(module_idx, offset.0 as usize)?.func(machine, module_idx)
                    },

                    ExternFunc::Host(host_fn) => todo!(),
                }
            },
        }
    }

    pub(crate) fn new(ty: TypeIdx, code_idx: CodeIdx) -> Self {
        Self {
            r#type: ty,
            r#impl: FuncInstImpl::Local(code_idx)
        }
    }

    #[inline]
    pub(crate) fn typeidx(&self) -> &TypeIdx {
        &self.r#type
    }
}

