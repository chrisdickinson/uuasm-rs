use std::{sync::Arc, collections::HashMap};

use super::{value::Value, module::Module, TKTK};

type GuestIndex = usize;
type ExternKey = (usize, usize);

#[derive(Debug)]
enum ExternGlobal<'a> {
    Host(&'a mut Value),
    Guest(GuestIndex, usize), 
}

#[derive(Debug)]
enum Extern<'a> {
    Func(TKTK),
    Global(ExternGlobal<'a>),
    Table(TKTK),
    Memory(TKTK),
    SharedMemory(TKTK),
}

#[derive(Debug, Default)]
pub(crate) struct Imports<'a> {
    globals: Vec<Value>,

    // borrow a page from wasmtime!
    strings: Vec<Arc<str>>,
    string_to_idx: HashMap<Arc<str>, usize>,
    guests: Vec<Module<'a>>,
    externs: HashMap<ExternKey, Extern<'a>>,
}

impl<'a> Imports<'a> {
    pub(crate) fn new(globals: Vec<Value>) -> Self {
        Self {
            globals,
            ..Default::default()
        }
    }

    fn intern(&mut self, string: &str) -> usize {
        if let Some(idx) = self.string_to_idx.get(string) {
            return *idx
        }

        let string: Arc<str> = string.into();
        let idx = self.strings.len();
        self.strings.push(string.clone());
        self.string_to_idx.insert(string, idx);
        idx
    }
}


