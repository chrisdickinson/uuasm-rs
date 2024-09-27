use std::{
    collections::{HashMap, LinkedList},
    marker::Sized,
    num::NonZeroU64,
};

use uuasm_ir::{FuncIdx, NumType, RefType, ValType, VecType};

use crate::{value::RefValue, Value};

// Try to land each stack segment in 65355 bytes of memory (two pointers for
// the linked list next/prev, plus the overhead of the bookkeeping in the struct.)
const STACK_SEGMENT_PAGE_SIZE: usize =
    0x10000 - (size_of::<StackSegment<0>>() + (size_of::<*const ()>() * 2));

pub(crate) type PageSizedStack = SegmentedStack<STACK_SEGMENT_PAGE_SIZE>;

#[inline]
#[cold]
fn cold() {}

pub trait Pushable {}

impl Pushable for i32 {}
impl Pushable for i64 {}
impl Pushable for u32 {}
impl Pushable for u64 {}
impl Pushable for f32 {}
impl Pushable for f64 {}
impl Pushable for i128 {}
impl Pushable for RefValue {}

pub(crate) trait Stack {
    type Fence;

    fn write_at_fence<T: Sized + Copy>(&mut self, fence: &Self::Fence, val: T);
    fn read_at_fence<T: Sized + Copy>(&self, fence: &Self::Fence) -> T;
    fn pop<T: Sized + Copy>(&mut self) -> T;
    fn push<T: Sized + Copy + Pushable>(&mut self, val: T);
    fn push_value(&mut self, val: StackValue);
    fn pop_valtype(&mut self, val_type: ValType) -> StackValue;

    fn fence(&self) -> Self::Fence;
    fn unwind(&mut self, fence: Self::Fence);
}

pub(crate) struct SegmentedStack<const N: usize> {
    stack: LinkedList<StackSegment<N>>,
    depth: u32,
}

#[derive(Clone)]
pub(crate) enum StackValue {
    I32(i32),
    I64(i64),
    F32(u32),
    F64(u64),

    // Box up i128 because it's infrequently used in block types / selects / etc
    V128(Box<i128>),
    RefFunc(RefValue),
    RefExtern(RefValue),
}

impl StackValue {
    pub fn as_i32(&self) -> Option<i32> {
        if let Self::I32(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    pub fn as_ref_value(&self) -> RefValue {
        if let Self::RefFunc(v) | Self::RefExtern(v) = self {
            *v
        } else {
            None
        }
    }
}

impl From<Value> for StackValue {
    fn from(value: Value) -> Self {
        match value {
            Value::I32(xs) => StackValue::I32(xs),
            Value::I64(xs) => StackValue::I64(xs),
            Value::F32(xs) => StackValue::F32(xs.to_bits()),
            Value::F64(xs) => StackValue::F64(xs.to_bits()),
            Value::V128(xs) => StackValue::V128(Box::new(xs)),
            Value::RefNull => StackValue::RefFunc(None),
            Value::RefExtern(xs) => StackValue::RefExtern(NonZeroU64::new(xs as u64 + 1)),
            _ => panic!("cannot coerce that value into stack value"),
        }
    }
}

impl From<StackValue> for Value {
    fn from(value: StackValue) -> Self {
        match value {
            StackValue::I32(xs) => Self::I32(xs),
            StackValue::I64(xs) => Self::I64(xs),
            StackValue::F32(xs) => Self::F32(f32::from_bits(xs)),
            StackValue::F64(xs) => Self::F64(f64::from_bits(xs)),
            StackValue::V128(xs) => Self::V128(*xs),
            StackValue::RefFunc(None) => Self::RefNull,
            StackValue::RefExtern(None) => Self::RefNull,
            StackValue::RefFunc(Some(xs)) => Self::RefFunc(FuncIdx(xs.get() as u32)),
            StackValue::RefExtern(Some(xs)) => Self::RefExtern(xs.get() as u32 - 1),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Fence(u32, u16);

// TODO: we need a way to unwind without having to re-specify the exact types
// (e.g., in the case of "br" instructions)

impl<const N: usize> SegmentedStack<N> {
    pub(crate) fn new() -> Self {
        Self {
            stack: LinkedList::new(),
            depth: 0,
        }
    }
}

impl<const N: usize> Stack for SegmentedStack<N> {
    type Fence = Fence;

    fn fence(&self) -> Fence {
        Fence(
            self.depth,
            self.stack.front().map(|xs| xs.ptr).unwrap_or_default(),
        )
    }

    fn unwind(&mut self, fence: Fence) {
        let Fence(depth, ptr) = fence;
        for _ in depth..self.depth {
            let _ = self.stack.pop_front();
        }

        let Some(segment) = self.stack.front_mut() else {
            if depth == 0 && ptr == 0 {
                return;
            }
            panic!("received invalid fence");
        };
        segment.unwind(ptr);
    }

    fn push<T: Sized + Copy + Pushable>(&mut self, val: T) {
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

    fn pop<T: Sized + Copy>(&mut self) -> T {
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

    fn push_value(&mut self, val: StackValue) {
        match val {
            StackValue::I32(xs) => self.push(xs),
            StackValue::I64(xs) => self.push(xs),
            StackValue::F32(xs) => self.push(xs),
            StackValue::F64(xs) => self.push(xs),
            StackValue::V128(xs) => self.push(*xs),
            StackValue::RefFunc(xs) => self.push(xs),
            StackValue::RefExtern(xs) => self.push(xs),
        }
    }

    fn pop_valtype(&mut self, val_type: ValType) -> StackValue {
        match val_type {
            ValType::NumType(NumType::I32) => StackValue::I32(self.pop::<i32>()),
            ValType::NumType(NumType::F32) => StackValue::F32(self.pop::<u32>()),
            ValType::NumType(NumType::I64) => StackValue::I64(self.pop::<i64>()),
            ValType::NumType(NumType::F64) => StackValue::F64(self.pop::<u64>()),
            ValType::VecType(VecType::V128) => StackValue::V128(Box::new(self.pop::<i128>())),
            ValType::RefType(RefType::FuncRef) => StackValue::RefFunc(self.pop::<RefValue>()),
            ValType::RefType(RefType::ExternRef) => StackValue::RefExtern(self.pop::<RefValue>()),
            ValType::Never => unreachable!(),
        }
    }

    fn write_at_fence<T: Sized + Copy>(&mut self, fence: &Fence, val: T) {
        let Fence(depth, ptr) = fence;
        let Some(segment) = self.stack.iter_mut().nth(*depth as usize) else {
            cold();
            panic!("invalid fence");
        };

        segment.write_at(*ptr, val);
    }

    fn read_at_fence<T: Sized + Copy>(&self, fence: &Fence) -> T {
        let Fence(depth, ptr) = fence;
        let Some(segment) = self.stack.iter().nth(*depth as usize) else {
            cold();
            panic!("invalid fence");
        };

        segment.read_at(*ptr)
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

    pub(crate) fn push<T: Sized + Pushable>(&mut self, val: T) {
        let size = size_of::<T>();
        let align = align_of::<T>();

        let misalignment = unsafe { self.storage.as_ptr().add(self.ptr as usize) as usize } % align;
        let adjustment = align - misalignment;
        #[cfg(any())]
        eprintln!(
            "push: {:x?} size={size} align={align} misalignment={misalignment}; typename={}",
            unsafe { self.storage.as_ptr().add(self.ptr as usize) },
            std::any::type_name::<T>(),
        );

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
        #[cfg(any())]
        eprintln!(
            "pop: {:x?} size={size} padding={padding}; typename={}",
            unsafe {
                self.storage
                    .as_ptr()
                    .add(self.ptr as usize - size - padding as usize)
            },
            std::any::type_name::<T>(),
        );

        let ptr = self.ptr;
        self.ptr -= size as u16 + padding as u16;
        let buf_ref = &self.storage[(ptr as usize) - size..(ptr as usize)];

        let value = buf_ref.as_ptr() as *const T;
        unsafe { *value }
    }

    pub(crate) fn read_at<T: Sized + Copy>(&self, ptr: u16) -> T {
        let size = size_of::<T>();

        let ptr_lower = ptr - size as u16;
        let buf_ref = &self.storage[(ptr_lower as usize)..(ptr as usize)];

        let value = buf_ref.as_ptr() as *const T;
        unsafe { *value }
    }

    pub(crate) fn write_at<T: Sized + Copy>(&mut self, ptr: u16, value: T) {
        let valptr = std::ptr::from_ref(&value);
        let count = size_of::<T>();
        let ptr_lower = ptr - count as u16;

        unsafe {
            std::ptr::copy_nonoverlapping(
                valptr,
                self.storage[(ptr_lower as usize)..(ptr as usize)].as_mut_ptr() as *mut T,
                1,
            );
        }
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
    impl Pushable for ExamplePointerInfo {}

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
        let mut stack = SegmentedStack::<0x40>::new();
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
        let mut stack = SegmentedStack::<0x40>::new();
        stack.push(13i32);
        let fence = stack.fence();
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
        let mut stack = SegmentedStack::<0x30>::new();
        stack.push(13i32);
        stack.push(-1000000i32);
        stack.push(ExamplePointerInfo {
            module_idx: 0xffff_0000,
            func_idx: 0xff00_ff00,
        });
        stack.push(0x0800_8080i32);
        stack.push(1i64);
        let fence = stack.fence();
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
