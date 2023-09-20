#![allow(dead_code)]

#![allow(clippy::upper_case_acronyms)]

use std::sync::Arc;
use std::alloc::{ self, Layout };
use rustix::mm::MprotectFlags;

use crate::nodes::{Func, TypeIdx, Module, Code, FuncType, Import, ImportDesc, TableType, MemType};

const PAGE_SIZE: usize = 1 << 16;

struct TKTK;

struct Stack {
    stack: Vec<StackItem>,
}

enum StackItem {
    Value,
    Label,
    Activation,
}

#[derive(Debug)]
enum Value {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    V128(i128),
    RefNull(Box<Value>),
    RefFunc(FuncAddr),
    RefExtern(ExternAddr),
}

enum FuncInstImpl {
    Guest(Arc<ModuleInstance>, TypeIdx, Func),
    Host(TKTK)
}

struct FuncInst {
    r#type: FuncType,
    r#impl: FuncInstImpl,
}

#[derive(Debug)]
struct TableInst(TableType, Vec<Value>);

#[derive(Debug)]
struct GuardedRegion {
    bytes: *mut u8,
    layout: Layout
}

impl GuardedRegion {
    fn new(page_count: usize) -> Self {
        let layout = Layout::from_size_align(page_count * PAGE_SIZE, PAGE_SIZE).unwrap();
        let bytes = unsafe { alloc::alloc(layout) };

        let bottom_of_stack = unsafe { bytes.add(page_count * PAGE_SIZE) };

        // TODO: well, this is almost certainly incorrect.
        // tested using
        // ```
        // let reference = unsafe { &mut *(bytes.add(page_count * PAGE_SIZE)) };
        // dbg!(reference);
        // ```
        // after the mprotect call -- if the mprotect isn't set the test passes (incorrectly),
        // with the mprotect we get SIGBUS (as expected.)
        //
        // amongst the things that aren't right, the one I know about is that we're not protecting
        // that whole 8GiB address space above the stack.
        unsafe { rustix::mm::mprotect(bottom_of_stack.cast(), PAGE_SIZE, MprotectFlags::empty())
            .expect("failed to protect stack guard page"); };


        GuardedRegion { bytes, layout }
    }

    fn grow(_page_count: usize) {
        todo!("not yet implemented")
    }
}

impl Drop for GuardedRegion {
    fn drop(&mut self) {
        unsafe { alloc::dealloc(self.bytes, self.layout) }
    }
}

#[derive(Debug)]
struct MemInst(MemType, GuardedRegion);
type GlobalInst = TKTK;
type ElemInst = TKTK;
type DataInst = TKTK;
type ExportInst = TKTK;

// This should really be an ECS store.
#[derive(Debug)]
struct Store {
    func_protos: Vec<FuncProto>,

    // funcs: Vec<FuncInst>,
    tables: Vec<TableInst>,
    memories: Vec<MemInst>,
    // globals: Vec<GlobalInst>,
    // elems: Vec<ElemInst>,
    // datas: Vec<DataInst>
}

#[derive(Debug)]
struct GuestFuncProto {
    func_type: FuncType,
    func: Func
}

#[derive(Debug)]
struct HostFuncProto {
    func_type: FuncType,
    offset: u32
}

#[derive(Debug)]
enum FuncProto {
    Host(HostFuncProto),
    Guest(GuestFuncProto),
}

enum Extern {
    Func(TKTK),
    Global(TKTK),
    Table(TKTK),
    Memory(TKTK),
    SharedMemory(TKTK),
}

impl Store {
    fn new(module: &Module) -> anyhow::Result<Self> {
        let type_sections = module.type_section();
        let import_sections = module.import_section();
        let function_sections = module.function_section();
        let table_sections = module.table_section();
        let memory_sections = module.memory_section();
        let global_sections = module.global_section();
        let export_sections = module.export_section();
        let start_sections = module.start_section();
        let element_sections = module.element_section();
        let code_sections = module.code_section();
        let data_sections = module.data_section();
        let datacount_sections = module.datacount_section();

        // NOTE: this is inexact -- not every import is a function -- but it's
        // a workable upper bound.
        let mut func_protos = Vec::with_capacity(
            import_sections.map(|xs| xs.len()).unwrap_or(0) +
            code_sections.map(|xs| xs.len()).unwrap_or(0)
        );
        let mut import_offset = 0;

        if let Some((imports, types)) = import_sections.zip(type_sections) {
            for import in imports {
                match import.desc {
                    ImportDesc::Func(type_idx) => {
                        let Some(func_type) = types.get(type_idx.0 as usize) else {
                            anyhow::bail!("host function type ({}) out of bounds (len={})", type_idx.0, types.len());
                        };

                        let func_type = func_type.clone();
                        func_protos.push(FuncProto::Host(HostFuncProto { func_type, offset: import_offset }));
                        import_offset += 1;
                    },

                    ImportDesc::Table(_) => todo!(),
                    ImportDesc::Mem(_) => todo!(),
                    ImportDesc::Global(_) => todo!(),
                }
            }
        }

        if let Some(((codes, func_type_indices), types)) = code_sections.zip(function_sections).zip(type_sections) {
            let length = codes.len();
            if func_type_indices.len() != length {
                anyhow::bail!("function section length ({}) did not match code section length ({})", func_type_indices.len(), length);
            }

            for (code, type_idx) in codes.iter().zip(func_type_indices.iter()) {
                let Code(func) = code;
                let func = func.clone();

                let Some(func_type) = types.get(type_idx.0 as usize) else {
                    anyhow::bail!("guest function type ({}) out of bounds (len={})", type_idx.0, types.len());
                };

                let func_type = func_type.clone();
                func_protos.push(FuncProto::Guest(GuestFuncProto { func_type, func }));
            }
        }

        let mut tables = Vec::with_capacity(16);
        for table_type in table_sections.into_iter().flatten() {
            let v = Vec::with_capacity(table_type.1.min() as usize);
            tables.push(TableInst(*table_type, v));
        }

        let mut memories = Vec::with_capacity(1);
        for memory_type in memory_sections.into_iter().flatten() {
            let lower_bound_pages = memory_type.0.min();
            let region = GuardedRegion::new(lower_bound_pages as usize);

            memories.push(MemInst(*memory_type, region));
        }

        Ok(Store {
            func_protos,
            tables,
            memories
        })
    }

    fn instantiate(&self, externs: &[Extern]) -> ModuleInstance {
        todo!();
    }

}

#[repr(transparent)]
#[derive(Debug)]
struct Addr(u32);

type FuncAddr = Addr;
type TableAddr = Addr;
type MemAddr = Addr;
type GlobalAddr = Addr;
type ElemAddr = Addr;
type DataAddr = Addr;
type ExternAddr = Addr;

struct ModuleInstance {
    types: Vec<FuncType>,
    funcaddrs: Vec<FuncAddr>,
    tableaddrs: Vec<TableAddr>,
    memaddrs: Vec<MemAddr>,
    globaladdrs: Vec<GlobalAddr>,
    elemaddrs: Vec<ElemAddr>,
    dataaddrs: Vec<DataAddr>,
    exports: Vec<ExportInst>,
}

#[cfg(test)]
mod test {
    use crate::parse::parse;
    use super::*;

    #[test]
    fn test_create_store() {
        let bytes = include_bytes!("../example.wasm");

        let wasm = parse(bytes).unwrap();

        let xs = Store::new(&wasm).unwrap();

        dbg!(xs);
    }
}
