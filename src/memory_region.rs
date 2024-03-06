use std::{
    alloc::{self, Layout},
    slice::{from_raw_parts, from_raw_parts_mut},
};

use crate::nodes::Limits;

const PAGE_SHIFT: usize = 16;
const PAGE_SIZE: usize = 1 << PAGE_SHIFT;
const LAST_PAGE: usize = 0xffff_ffff >> PAGE_SHIFT;

#[derive(Debug)]
pub(crate) struct MemoryRegion {
    bytes: *mut u8,
    page_count: usize,
    layout: Layout,
    limits: (usize, usize),
}

impl MemoryRegion {
    pub(crate) fn new(limits: Limits) -> Self {
        let page_count = limits.min() as usize;
        let max_page_count = limits.max().map(|xs| xs as usize).unwrap_or(0xffff_ffff);
        let layout = Layout::from_size_align(page_count << PAGE_SHIFT, PAGE_SIZE).unwrap();
        let bytes = unsafe { alloc::alloc(layout) };

        MemoryRegion {
            bytes,
            page_count,
            layout,
            limits: (page_count, max_page_count),
        }
    }

    pub fn len(&self) -> usize {
        return self.page_count << PAGE_SHIFT;
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        unsafe { from_raw_parts(self.bytes, self.page_count << PAGE_SHIFT) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { from_raw_parts_mut(self.bytes, self.page_count << PAGE_SHIFT) }
    }

    pub(crate) fn write<const U: usize>(&self, addr: usize, value: &[u8; U]) {
        unsafe { std::ptr::copy_nonoverlapping(value.as_ptr(), self.bytes.wrapping_add(addr), U); }
    }

    pub(crate) fn copy_data(&mut self, data: &[u8], offset: usize) -> anyhow::Result<()> {
        let desired_size = offset + data.len();
        let desired_pages = desired_size >> PAGE_SHIFT;
        self.grow(desired_pages)?;

        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.bytes.add(offset), data.len()) };

        Ok(())
    }

    pub(crate) fn grow(&mut self, page_count: usize) -> anyhow::Result<usize> {
        if page_count <= self.page_count {
            return Ok(self.page_count);
        }

        if page_count >= self.limits.1 {
            anyhow::bail!(
                "cannot allocate more than max limit of memory ({} pages)",
                self.limits.1
            );
        }

        if page_count >= LAST_PAGE {
            anyhow::bail!("cannot allocate more than 4GiB of memory");
        }

        self.layout = Layout::from_size_align(page_count << PAGE_SHIFT, PAGE_SIZE).unwrap();
        self.bytes = unsafe { alloc::realloc(self.bytes, self.layout, page_count << PAGE_SHIFT) };
        let old_page_count = self.page_count;
        self.page_count = page_count;
        Ok(old_page_count)
    }
}

impl Drop for MemoryRegion {
    fn drop(&mut self) {
        unsafe { alloc::dealloc(self.bytes, self.layout) };
    }
}
