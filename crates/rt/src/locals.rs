use std::{collections::LinkedList, iter::repeat};

use smallvec::SmallVec;
use uuasm_ir::{Local, NumType, RefType, ValType, VecType};

use crate::{
    prelude::ValTypeExtras,
    stack::{Stack, StackValue},
    value::RefValue,
};

pub(crate) struct StackMapStack<T: Stack> {
    storage: LinkedList<StackMap<T>>,
}

fn size_of_typelist(types: &[ValType]) -> usize {
    let mut size = 0;
    for t in types {
        size += match t {
            ValType::NumType(NumType::I32 | NumType::F32) => 4,
            ValType::NumType(NumType::I64 | NumType::F64) => 8,
            ValType::VecType(_) => 16,
            ValType::RefType(_) => size_of::<RefValue>(),
            ValType::Never => 0,
        }
    }
    size
}

impl<T: Stack> StackMapStack<T> {
    pub(crate) fn new() -> Self {
        Self {
            storage: LinkedList::new(),
        }
    }

    pub(crate) fn begin_call(
        &mut self,
        storage: &mut T,
        param_types: &[ValType],
        locals: &[Local],
    ) {
        let stack_values: SmallVec<[StackValue; 8]> = param_types
            .iter()
            .rev()
            .map(|val_type| storage.pop_valtype(*val_type))
            .collect();

        let first_fence = storage.fence();
        let locals: SmallVec<[(_, _); 8]> = stack_values
            .into_iter()
            .rev()
            .enumerate()
            .map(|(idx, value)| (unsafe { *param_types.get_unchecked(idx) }, value))
            .chain(locals.iter().flat_map(|Local(count, val_type)| {
                repeat((*val_type, val_type.instantiate_stackvalue())).take(*count as usize)
            }))
            .map(|(val_type, value)| {
                storage.push_value(value);
                (val_type, storage.fence())
            })
            .collect();

        self.storage.push_front(StackMap {
            first_fence,
            locals,
        })
    }

    pub(crate) fn end_call(&mut self, storage: &mut T, return_types: &[ValType]) {
        let Some(map) = self.storage.pop_front() else {
            panic!("empty stack");
        };
        let stack_values: SmallVec<[StackValue; 8]> = return_types
            .iter()
            .rev()
            .map(|val_type| storage.pop_valtype(*val_type))
            .collect();
        storage.unwind(map.first_fence);
        for stack_value in stack_values.into_iter().rev() {
            storage.push_value(stack_value);
        }
    }

    pub(crate) fn get(&self, storage: &mut T, idx: u32) {
        let front = self.storage.front().unwrap();
        front.get(storage, idx)
    }

    pub(crate) fn set(&mut self, storage: &mut T, idx: u32) {
        let front = self.storage.front_mut().unwrap();
        front.set(storage, idx)
    }

    pub(crate) fn tee(&mut self, storage: &mut T, idx: u32) {
        let front = self.storage.front_mut().unwrap();
        front.tee(storage, idx)
    }
}

// Locals work by pushing an item onto the stack, then creating a fence.
// The fence can then be used as part of a `write_at` or `read_at` call
// against the value stack
struct StackMap<T: Stack> {
    /// A map of `local index` -> type and offset information.
    locals: SmallVec<[(ValType, T::Fence); 8]>,
    first_fence: T::Fence,
}

impl<T: Stack> StackMap<T> {
    fn get(&self, storage: &mut T, idx: u32) {
        let (val_type, fence) = unsafe { self.locals.get_unchecked(idx as usize) };
        match val_type {
            ValType::NumType(NumType::I32 | NumType::F32) => {
                storage.push(storage.read_at_fence::<u32>(fence))
            }
            ValType::NumType(NumType::I64 | NumType::F64) => {
                storage.push(storage.read_at_fence::<u64>(fence))
            }
            ValType::VecType(VecType::V128) => {
                storage.push(storage.read_at_fence::<[u8; 16]>(fence))
            }
            ValType::RefType(RefType::FuncRef | RefType::ExternRef) => {
                storage.push(storage.read_at_fence::<RefValue>(fence))
            }
            ValType::Never => unreachable!(),
        }
    }

    fn set(&self, storage: &mut T, idx: u32) {
        let (val_type, fence) = unsafe { self.locals.get_unchecked(idx as usize) };
        match val_type {
            ValType::NumType(NumType::I32) => {
                let value = storage.pop::<i32>();
                storage.write_at_fence(fence, value);
            }
            ValType::NumType(NumType::F32) => {
                let value = storage.pop::<f32>();
                storage.write_at_fence(fence, value);
            }
            ValType::NumType(NumType::I64) => {
                let value = storage.pop::<i64>();
                storage.write_at_fence(fence, value);
            }
            ValType::NumType(NumType::F64) => {
                let value = storage.pop::<f64>();
                storage.write_at_fence(fence, value);
            }
            ValType::VecType(VecType::V128) => {
                let value = storage.pop::<[u8; 16]>();
                storage.write_at_fence(fence, value);
            }
            ValType::RefType(RefType::FuncRef) => {
                let value = storage.pop::<RefValue>();
                storage.write_at_fence(fence, value);
            }
            ValType::RefType(RefType::ExternRef) => {
                let value = storage.pop::<RefValue>();
                storage.write_at_fence(fence, value);
            }
            ValType::Never => unreachable!(),
        }
    }

    fn tee(&self, storage: &mut T, idx: u32) {
        let (val_type, fence) = unsafe { self.locals.get_unchecked(idx as usize) };
        match val_type {
            ValType::NumType(NumType::I32) => {
                let value = storage.pop::<i32>();
                storage.write_at_fence(fence, value);
                storage.push(value);
            }
            ValType::NumType(NumType::F32) => {
                let value = storage.pop::<f32>();
                storage.write_at_fence(fence, value);
                storage.push(value);
            }
            ValType::NumType(NumType::I64) => {
                let value = storage.pop::<i64>();
                storage.write_at_fence(fence, value);
                storage.push(value);
            }
            ValType::NumType(NumType::F64) => {
                let value = storage.pop::<f64>();
                storage.write_at_fence(fence, value);
                storage.push(value);
            }
            ValType::VecType(VecType::V128) => {
                let value = storage.pop::<[u8; 16]>();
                storage.write_at_fence(fence, value);
                storage.push(value);
            }
            ValType::RefType(RefType::FuncRef) => {
                let value = storage.pop::<RefValue>();
                storage.write_at_fence(fence, value);
                storage.push(value);
            }
            ValType::RefType(RefType::ExternRef) => {
                let value = storage.pop::<RefValue>();
                storage.write_at_fence(fence, value);
                storage.push(value);
            }
            ValType::Never => unreachable!(),
        }
    }
}
