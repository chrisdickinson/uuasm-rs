use std::{sync::Arc, collections::HashMap};

use crate::{nodes::{Instr, BlockType}, memory_region::MemoryRegion};

use super::{instance::ModuleInstance, value::Value, Module, imports::{ExternKey, Extern}, Imports};

type InstanceIdx = usize;

struct Frame<'a> {
    #[cfg(test)]
    name: &'static str,
    pc: usize,
    return_unwind_count: usize,
    instrs: &'a [Instr],
    jump_to: Option<usize>,
    block_type: BlockType,
    locals_base_offset: usize,
    instance: InstanceIdx,
}

pub(super) struct Machine<'a> {
    modules: Box<[Module<'a>]>,
    instances: Box<[ModuleInstance]>,

    memories: Vec<MemoryRegion>,
    tables: Vec<Vec<Value>>,

    locals: Vec<Value>,
    globals: Vec<Value>,
    value_stack: Vec<Value>,
    frame_stack: Vec<Frame<'a>>,

    strings: Box<[Arc<str>]>,
    externs: HashMap<ExternKey, Extern>,
}

impl<'a> Machine<'a> {
   pub(super) fn new(imports: Imports<'a>, instances: Vec<ModuleInstance>) -> Self {
       Self {
           instances: instances.into_boxed_slice(),
           locals: vec![],
           globals: vec![],
           value_stack: vec![],
           frame_stack: vec![],

           strings: imports.strings.into_boxed_slice(),
           modules: imports.guests.into_boxed_slice(),
           memories: vec![],
           externs: todo!(),
       }
   }
}
