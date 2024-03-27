use crate::nodes::{Instr, BlockType};

use super::{instance::ModuleInstance, value::Value};

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

struct Machine<'a> {
    instances: Box<[ModuleInstance<'a>]>,
    locals: Vec<Value>,
    globals: Vec<Value>,
    value_stack: Vec<Value>,
    frame_stack: Vec<Frame<'a>>,
}

impl<'a> Machine<'a> {
    
}
