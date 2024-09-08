use std::{collections::LinkedList, marker::Sized};

#[inline]
#[cold]
fn cold() {}

pub(crate) struct Stack<const N: usize> {
    stack: LinkedList<StackSegment<N>>,
}

impl<const N: usize> Stack<N> {
    pub(crate) fn new() -> Self {
        Self {
            stack: LinkedList::new(),
        }
    }

    pub(crate) fn push<T: Sized + Copy>(&mut self, val: T) {
        let Some(head) = self.stack.front_mut() else {
            cold();
            self.stack.push_front(StackSegment::new());
            return self.push(val);
        };

        if !head.fits::<T>() {
            cold();
            self.stack.push_front(StackSegment::new());
            return self.push(val);
        }

        head.push(val)
    }

    pub(crate) fn pop<T: Sized + Copy>(&mut self) -> T {
        if let Some(head) = self.stack.front_mut() {
            if !head.is_empty() {
                return head.pop();
            } else {
                cold();
            }
        } else {
            cold();
            panic!("empty head");
        }

        self.stack.pop_front();
        self.pop()
    }
}

pub(crate) struct StackSegment<const N: usize> {
    storage: [u8; N],
    alignments: Vec<u8>,
    ptr: usize,
}

impl<const N: usize> StackSegment<N> {
    pub(crate) fn new() -> Self {
        Self {
            storage: [0; N],
            alignments: vec![],
            ptr: 0,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.ptr == 0
    }

    pub(crate) fn fits<T: Sized + Copy>(&self) -> bool {
        let size = size_of::<T>();
        let align = align_of::<T>();

        let misalignment = unsafe { self.storage.as_ptr().add(self.ptr) as usize } % align;
        let adjustment = align - misalignment;

        (self.ptr + adjustment + size) < N
    }

    pub(crate) fn push<T: Sized>(&mut self, val: T) {
        let size = size_of::<T>();
        let align = align_of::<T>();

        let misalignment = unsafe { self.storage.as_ptr().add(self.ptr) as usize } % align;
        let adjustment = align - misalignment;

        self.alignments.push(adjustment as u8);
        self.ptr += adjustment;

        let valptr = std::ptr::from_ref(&val);

        unsafe {
            std::ptr::copy_nonoverlapping(
                valptr,
                self.storage[self.ptr..self.ptr + size].as_mut_ptr() as *mut T,
                1,
            );
        }
        self.ptr += size;
    }

    pub(crate) fn pop<T: Sized + Copy>(&mut self) -> T {
        let size = size_of::<T>();
        let Some(padding) = self.alignments.pop() else {
            unreachable!();
        };

        let ptr = self.ptr;
        self.ptr -= size + padding as usize;
        let buf_ref = &self.storage[ptr - size..ptr];

        let value = buf_ref.as_ptr() as *const T;
        unsafe { *value }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_stack_segment() -> Result<(), ()> {
        let mut stack = StackSegment::<0xffff>::new();

        #[derive(Debug, Clone, Copy)]
        struct PointerInfo {
            module_idx: u32,
            func_idx: u32,
        }

        stack.push(13i32);
        stack.push(-1000000i32);
        stack.push(PointerInfo {
            module_idx: 0xffff_0000,
            func_idx: 0xff00_ff00,
        });
        stack.push(213i32);
        stack.push(line!());
        stack.push(1i64);
        stack.push(2i128);

        dbg!(stack.pop::<i128>());
        dbg!(stack.pop::<i64>());
        dbg!(stack.pop::<u32>());
        dbg!(stack.pop::<i32>());
        dbg!(stack.pop::<PointerInfo>());
        dbg!(stack.pop::<i32>());
        dbg!(stack.pop::<i32>());

        Ok(())
    }

    #[test]
    fn test_stack() -> Result<(), ()> {
        let mut stack = Stack::<0x20>::new();

        #[derive(Debug, Clone, Copy)]
        struct PointerInfo {
            module_idx: u32,
            func_idx: u32,
        }

        stack.push(13i32);
        stack.push(-1000000i32);
        stack.push(PointerInfo {
            module_idx: 0xffff_0000,
            func_idx: 0xff00_ff00,
        });
        stack.push(213i32);
        stack.push(line!());
        stack.push(1i64);
        stack.push(2i128);

        dbg!(stack.pop::<i128>());
        dbg!(stack.pop::<i64>());
        dbg!(stack.pop::<u32>());
        dbg!(stack.pop::<i32>());
        dbg!(stack.pop::<PointerInfo>());
        dbg!(stack.pop::<i32>());
        dbg!(stack.pop::<i32>());

        Ok(())
    }
}
