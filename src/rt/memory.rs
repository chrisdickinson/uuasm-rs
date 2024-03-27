use crate::{memory_region::MemoryRegion, nodes::{MemType, ByteVec, Import}};

use super::{TKTK, imports::Imports};

#[derive(Debug)]
enum MemoryInstImpl {
    Guest(MemoryRegion),
    Host(TKTK),
}

#[derive(Debug)]
pub(crate) struct MemInst {
    r#type: MemType,
    r#impl: MemoryInstImpl,
}

impl MemInst {
    pub(crate) fn resolve(ty: MemType, import: &Import<'_>, imports: &Imports) -> anyhow::Result<Self> {
        todo!()
    }

    pub(crate) fn new(ty: MemType) -> Self {
        Self {
            r#type: ty,
            r#impl: MemoryInstImpl::Guest(MemoryRegion::new(ty.0)),
        }
    }

    pub(crate) fn copy_data(&mut self, byte_vec: &ByteVec<'_>, offset: usize) -> anyhow::Result<()> {
        match &mut self.r#impl {
            MemoryInstImpl::Guest(region) => region.copy_data(byte_vec.0, offset)?,
            MemoryInstImpl::Host(_tktk) => todo!(),
        }

        Ok(())
    }

    #[inline]
    pub(crate) fn grow(&mut self, page_count: usize) -> anyhow::Result<usize> {
        match &mut self.r#impl {
            MemoryInstImpl::Guest(memory) => {
                memory.grow(page_count)
            }
            MemoryInstImpl::Host(_) => todo!(),
        }
    }

    #[inline]
    pub(crate) fn load<const U: usize>(&self, addr: usize) -> anyhow::Result<[u8; U]> {
        match &self.r#impl {
            MemoryInstImpl::Guest(memory) => {
                if addr.saturating_add(U) > memory.len() {
                    anyhow::bail!("out of bounds memory access")
                }
                Ok(memory.as_slice()[addr..addr + U].try_into()?)
            }
            MemoryInstImpl::Host(_) => todo!(),
        }
    }

    #[inline]
    pub(crate) fn write<const U: usize>(&mut self, addr: usize, value: &[u8; U]) -> anyhow::Result<()> {
        match &self.r#impl {
            MemoryInstImpl::Guest(memory) => {
                if addr.saturating_add(U) > memory.len() {
                    anyhow::bail!("out of bounds memory access")
                }
                memory.write(addr, value);

                Ok(())
            }
            MemoryInstImpl::Host(_) => todo!(),
        }
    }
}
