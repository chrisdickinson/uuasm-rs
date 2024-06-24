use crate::nodes::{Import, MemIdx, MemType};

use super::{
    imports::{Extern, GuestIndex, LookupImport},
    machine::MachineMemoryIndex,
};

#[derive(Debug, Clone)]
pub(super) enum MemoryInstImpl {
    Local(MachineMemoryIndex),
    Remote(GuestIndex, MemIdx),
}

#[derive(Debug, Clone)]
pub(crate) struct MemInst {
    r#type: MemType,
    pub(super) r#impl: MemoryInstImpl,
}

impl MemInst {
    pub(crate) fn resolve(
        ty: MemType,
        import: &Import,
        imports: &impl LookupImport,
    ) -> anyhow::Result<Self> {
        let Some(ext) = imports.lookup(import) else {
            anyhow::bail!("could not resolve {}/{}", import.r#mod.0, import.nm.0);
        };

        let Extern::Memory(module_idx, mem_idx) = ext else {
            anyhow::bail!(
                "expected {}/{} to resolve to a memory",
                import.r#mod.0,
                import.nm.0
            );
        };

        Ok(Self {
            r#type: ty,
            r#impl: MemoryInstImpl::Remote(module_idx, mem_idx),
        })
    }

    pub(crate) fn new(ty: MemType, idx: MachineMemoryIndex) -> Self {
        Self {
            r#type: ty,
            r#impl: MemoryInstImpl::Local(idx),
        }
    }

    pub(crate) fn typedef(&self) -> &MemType {
        &self.r#type
    }

    /*
    pub(crate) fn copy_data(&mut self, byte_vec: &ByteVec, offset: usize) -> anyhow::Result<()> {
        match &mut self.r#impl {
            MemoryInstImpl::Local(region) => region.copy_data(byte_vec.0, offset)?,
            MemoryInstImpl::Remote(_tktk) => todo!(),
        }

        Ok(())
    }

    #[inline]
    pub(crate) fn grow(&mut self, page_count: usize) -> anyhow::Result<usize> {
        match &mut self.r#impl {
            MemoryInstImpl::Local(memory) => {
                memory.grow(page_count)
            }
            MemoryInstImpl::Remote(_) => todo!(),
        }
    }

    #[inline]
    pub(crate) fn load<const U: usize>(&self, addr: usize) -> anyhow::Result<[u8; U]> {
        match &self.r#impl {
            MemoryInstImpl::Local(memory) => {
                if addr.saturating_add(U) > memory.len() {
                    anyhow::bail!("out of bounds memory access")
                }
                Ok(memory.as_slice()[addr..addr + U].try_into()?)
            }
            MemoryInstImpl::Remote(_) => todo!(),
        }
    }

    #[inline]
    pub(crate) fn write<const U: usize>(&mut self, addr: usize, value: &[u8; U]) -> anyhow::Result<()> {
        match &self.r#impl {
            MemoryInstImpl::Local(memory) => {
                if addr.saturating_add(U) > memory.len() {
                    anyhow::bail!("out of bounds memory access")
                }
                memory.write(addr, value);

                Ok(())
            }
            MemoryInstImpl::Remote(_) => todo!(),
        }
    }
    */
}
