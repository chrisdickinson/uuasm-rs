use crate::nodes::Limits;

const PAGE_SHIFT: usize = 16;
const PAGE_SIZE: usize = 1 << PAGE_SHIFT;
const LAST_PAGE: usize = 0xffff_ffff >> PAGE_SHIFT;

#[derive(Debug)]
pub(crate) struct MemoryRegion {
    page_count: usize,
    limits: (usize, usize),
    storage: Vec<u8>,
}

impl MemoryRegion {
    pub(crate) fn new(limits: Limits) -> Self {
        let page_count = limits.min() as usize;
        let max_page_count = limits.max().map(|xs| xs as usize).unwrap_or(0xffff_ffff);
        let storage = vec![0u8; page_count << PAGE_SHIFT];

        MemoryRegion {
            storage,
            page_count,
            limits: (page_count, max_page_count),
        }
    }

    pub fn len(&self) -> usize {
        self.page_count << PAGE_SHIFT
    }

    pub fn page_count(&self) -> usize {
        self.page_count
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        self.storage.as_slice()
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        self.storage.as_mut_slice()
    }

    pub(crate) fn write<const U: usize>(&mut self, addr: usize, value: &[u8; U]) {
        self.storage[addr..addr.saturating_add(U)].copy_from_slice(value.as_slice());
    }

    pub(crate) fn fill_data(&mut self, val: u8, offset: usize, count: usize) -> anyhow::Result<()> {
        let desired_size = offset.saturating_add(count);
        if desired_size > self.len() {
            anyhow::bail!("out of bounds memory access");
        }

        self.storage[offset..count + offset].fill(val);

        Ok(())
    }

    // XXX: should we have an option for "just copy, do not grow"?
    pub(crate) fn copy_data(&mut self, data: &[u8], offset: usize) {
        self.storage[offset..offset + data.len()].copy_from_slice(data);
    }

    pub(crate) fn copy_overlapping_data(
        &mut self,
        offset: usize,
        from_offset: usize,
        count: usize,
    ) {
        self.storage
            .copy_within(from_offset..from_offset + count, offset);
    }

    pub(crate) fn grow_to_fit(&mut self, data: &[u8], offset: usize) -> anyhow::Result<usize> {
        let desired_size = offset + data.len();
        let desired_pages = desired_size >> PAGE_SHIFT;
        self.grow(desired_pages)
    }

    pub(crate) fn grow(&mut self, page_count: usize) -> anyhow::Result<usize> {
        let new_page_count = self.page_count + page_count;
        if new_page_count > self.limits.1 {
            return Ok(-1i32 as usize);
        }

        if new_page_count >= LAST_PAGE {
            anyhow::bail!("cannot allocate more than 4GiB of memory");
        }

        self.storage.resize(new_page_count << PAGE_SHIFT, 0);
        let old_page_count = self.page_count;
        self.page_count = new_page_count;
        Ok(old_page_count)
    }

    #[inline]
    pub(crate) fn load<const U: usize>(&self, addr: usize) -> anyhow::Result<[u8; U]> {
        if addr.saturating_add(U) > self.len() {
            anyhow::bail!("out of bounds memory access")
        }
        Ok(self.as_slice()[addr..addr + U].try_into()?)
    }

    #[inline]
    pub(crate) fn store<const U: usize>(
        &mut self,
        addr: usize,
        value: &[u8; U],
    ) -> anyhow::Result<()> {
        if addr.saturating_add(U) > self.len() {
            anyhow::bail!("out of bounds memory access")
        }
        self.write(addr, value);
        Ok(())
    }
}
