use std::{
    collections::{HashMap, LinkedList},
    marker::Sized,
};

// Try to land each stack segment in 65355 bytes of memory (two pointers for
// the linked list next/prev, plus the overhead of the bookkeeping in the struct.)
const STACK_SEGMENT_PAGE_SIZE: usize =
    0x10000 - (size_of::<StackSegment<0>>() + (size_of::<*const ()>() * 2));

pub(crate) type DefaultStack = Stack<STACK_SEGMENT_PAGE_SIZE>;

#[inline]
#[cold]
fn cold() {}

pub(crate) struct Stack<const N: usize> {
    stack: LinkedList<StackSegment<N>>,
    depth: u32,
}

#[derive(Debug)]
pub(crate) struct Fence(u32, u16);

// TODO: we need a way to unwind without having to re-specify the exact types
// (e.g., in the case of "br" instructions)

impl<const N: usize> Stack<N> {
    pub(crate) fn new() -> Self {
        Self {
            stack: LinkedList::new(),
            depth: 0,
        }
    }

    pub(crate) fn begin(&self) -> Fence {
        Fence(
            self.depth,
            self.stack.front().map(|xs| xs.ptr).unwrap_or_default(),
        )
    }

    pub(crate) fn unwind(&mut self, fence: Fence) {
        let Fence(depth, ptr) = fence;
        for _ in depth..self.depth {
            let _ = self.stack.pop_front();
        }

        let Some(segment) = self.stack.front_mut() else {
            cold();
            unreachable!("received invalid fence");
        };
        segment.unwind(ptr);
    }

    pub(crate) fn push<T: Sized + Copy + std::fmt::Debug>(&mut self, val: T) {
        let Some(head) = self.stack.front_mut() else {
            cold();
            self.stack.push_front(StackSegment::new());
            return self.push(val);
        };

        if !head.fits::<T>() {
            cold();
            self.depth += 1;
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
    alignments: HashMap<u16, u8>,
    ptr: u16,
}

impl<const N: usize> StackSegment<N> {
    pub(crate) fn new() -> Self {
        Self {
            storage: [0; N],
            alignments: HashMap::new(),
            ptr: 0,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.ptr == 0
    }

    pub(crate) fn fits<T: Sized + Copy>(&self) -> bool {
        let size = size_of::<T>();
        let align = align_of::<T>();

        let misalignment = unsafe { self.storage.as_ptr().add(self.ptr as usize) as usize } % align;
        let adjustment = align - misalignment;

        (self.ptr as usize + adjustment + size) < N
    }

    pub(crate) fn unwind(&mut self, to: u16) {
        // preserve any alignment keys that are <= our target.
        let alignments = self
            .alignments
            .iter()
            .filter_map(|(key, val)| if *key <= to { Some((*key, *val)) } else { None })
            .collect();
        self.alignments = alignments;
        self.ptr = to;
    }

    // TODO: pop_opaque(self, bytes) -> a series of slices + push_opaque(a series of slices)
    //       this would be handy for grabbing the last N values of the stack during a br instr
    //       and replacing them on the stack after rolling back to the br target

    pub(crate) fn push<T: Sized>(&mut self, val: T) {
        let size = size_of::<T>();
        let align = align_of::<T>();

        let misalignment = unsafe { self.storage.as_ptr().add(self.ptr as usize) as usize } % align;
        let adjustment = align - misalignment;

        self.ptr += adjustment as u16;

        let valptr = std::ptr::from_ref(&val);

        unsafe {
            std::ptr::copy_nonoverlapping(
                valptr,
                self.storage[(self.ptr as usize)..(self.ptr as usize) + size].as_mut_ptr()
                    as *mut T,
                1,
            );
        }
        self.ptr += size as u16;
        self.alignments.insert(self.ptr, adjustment as u8);
    }

    pub(crate) fn pop<T: Sized + Copy>(&mut self) -> T {
        let size = size_of::<T>();

        let padding = self.alignments.remove(&self.ptr).unwrap_or_default();

        let ptr = self.ptr;
        self.ptr -= size as u16 + padding as u16;
        let buf_ref = &self.storage[(ptr as usize) - size..(ptr as usize)];

        let value = buf_ref.as_ptr() as *const T;
        unsafe { *value }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct ExamplePointerInfo {
        module_idx: u32,
        func_idx: u32,
    }

    #[test]
    fn test_stack_segment() -> Result<(), ()> {
        let mut stack = StackSegment::<0xffff>::new();

        stack.push(13i32);
        stack.push(-1000000i32);
        stack.push(ExamplePointerInfo {
            module_idx: 0xffff_0000,
            func_idx: 0xff00_ff00,
        });
        stack.push(0x0800_8080i32);
        stack.push(1i64);
        stack.push(2i128);

        assert_eq!(stack.pop::<i128>(), 2);
        assert_eq!(stack.pop::<i64>(), 1);
        assert_eq!(stack.pop::<i32>(), 0x0800_8080);
        assert_eq!(
            stack.pop::<ExamplePointerInfo>(),
            ExamplePointerInfo {
                module_idx: 0xffff_0000,
                func_idx: 0xff00_ff00,
            }
        );
        assert_eq!(stack.pop::<i32>(), -1000000i32);
        assert_eq!(stack.pop::<i32>(), 13);
        Ok(())
    }

    #[test]
    fn test_stack() -> Result<(), ()> {
        let mut stack = Stack::<0x40>::new();
        stack.push(13i32);
        stack.push(-1000000i32);
        stack.push(ExamplePointerInfo {
            module_idx: 0xffff_0000,
            func_idx: 0xff00_ff00,
        });
        stack.push(0x0800_8080i32);
        stack.push(1i64);
        stack.push(2i128);

        assert_eq!(stack.pop::<i128>(), 2);
        assert_eq!(stack.pop::<i64>(), 1);
        assert_eq!(stack.pop::<i32>(), 0x0800_8080);
        assert_eq!(
            stack.pop::<ExamplePointerInfo>(),
            ExamplePointerInfo {
                module_idx: 0xffff_0000,
                func_idx: 0xff00_ff00,
            }
        );
        assert_eq!(stack.pop::<i32>(), -1000000i32);
        assert_eq!(stack.pop::<i32>(), 13);

        eprintln!(
            "size of default stack segment={:?}",
            size_of::<StackSegment<STACK_SEGMENT_PAGE_SIZE>>()
        );
        Ok(())
    }

    #[test]
    fn test_stack_unwind() -> Result<(), ()> {
        let mut stack = Stack::<0x40>::new();
        stack.push(13i32);
        let fence = stack.begin();
        stack.push(-1000000i32);
        stack.push(ExamplePointerInfo {
            module_idx: 0xffff_0000,
            func_idx: 0xff00_ff00,
        });
        stack.push(0x0800_8080i32);
        stack.push(1i64);
        stack.push(2i128);

        stack.unwind(fence);
        stack.push(0xdead_0000_beef_0000u64);

        assert_eq!(stack.pop::<u64>(), 0xdead_0000_beef_0000u64);
        assert_eq!(stack.pop::<i32>(), 13i32);

        Ok(())
    }

    #[test]
    fn test_stack_unwind_mid_frame() -> Result<(), ()> {
        let mut stack = Stack::<0x30>::new();
        stack.push(13i32);
        stack.push(-1000000i32);
        stack.push(ExamplePointerInfo {
            module_idx: 0xffff_0000,
            func_idx: 0xff00_ff00,
        });
        stack.push(0x0800_8080i32);
        stack.push(1i64);
        let fence = stack.begin();
        stack.push(2i128);
        stack.push(0xdead_0000_beef_0000u64);

        stack.unwind(fence);
        stack.push(0xe110_dadau32);

        assert_eq!(stack.pop::<u32>(), 0xe110_dadau32);
        assert_eq!(stack.pop::<i64>(), 1);
        assert_eq!(stack.pop::<i32>(), 0x0800_8080);
        assert_eq!(
            stack.pop::<ExamplePointerInfo>(),
            ExamplePointerInfo {
                module_idx: 0xffff_0000,
                func_idx: 0xff00_ff00,
            }
        );
        assert_eq!(stack.pop::<i32>(), -1000000i32);
        assert_eq!(stack.pop::<i32>(), 13);

        Ok(())
    }
}
