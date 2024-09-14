use std::{collections::LinkedList, iter::repeat};

use smallvec::SmallVec;
use uuasm_ir::{Local, NumType, RefType, ResultType, ValType, VecType};

use crate::{
    prelude::ValTypeExtras,
    stack::{Stack, StackValue},
    value::RefValue,
};

pub(crate) struct StackMapStack<T: Stack> {
    storage: LinkedList<StackMap<T>>,
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
        ResultType(param_types): &ResultType,
        locals: &[Local],
    ) {
        let stack_values: SmallVec<[StackValue; 8]> = param_types
            .iter()
            .rev()
            .map(|val_type| storage.pop_valtype(*val_type))
            .collect();

        let first_fence = storage.fence();
        let locals: Box<_> = stack_values
            .into_iter()
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

    pub(crate) fn end_call(&mut self, storage: &mut T, ResultType(return_types): &ResultType) {
        let Some(map) = self.storage.pop_front() else {
            panic!("empty stack");
        };
        let stack_values: SmallVec<[StackValue; 8]> = return_types
            .iter()
            .map(|val_type| storage.pop_valtype(*val_type))
            .collect();
        storage.unwind(map.first_fence);
        for stack_value in stack_values {
            storage.push_value(stack_value);
        }
    }

    pub(crate) fn get(&self, storage: &T, idx: u32) -> StackValue {
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
    locals: Box<[(ValType, T::Fence)]>,
    first_fence: T::Fence,
}

impl<T: Stack> StackMap<T> {
    fn get(&self, storage: &T, idx: u32) -> StackValue {
        let (val_type, fence) = unsafe { self.locals.get_unchecked(idx as usize) };
        match val_type {
            ValType::NumType(NumType::I32) => StackValue::I32(storage.read_at_fence(fence)),
            ValType::NumType(NumType::F32) => StackValue::F32(storage.read_at_fence(fence)),
            ValType::NumType(NumType::I64) => StackValue::I64(storage.read_at_fence(fence)),
            ValType::NumType(NumType::F64) => StackValue::F64(storage.read_at_fence(fence)),
            ValType::VecType(VecType::V128) => {
                StackValue::V128(Box::new(storage.read_at_fence(fence)))
            }
            ValType::RefType(RefType::FuncRef) => StackValue::RefFunc(storage.read_at_fence(fence)),
            ValType::RefType(RefType::ExternRef) => {
                StackValue::RefExtern(storage.read_at_fence(fence))
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
                let value = storage.pop::<i128>();
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
                let value = storage.pop::<i128>();
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
