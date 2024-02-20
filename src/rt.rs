#![allow(dead_code)]
#![allow(clippy::upper_case_acronyms)]

use rustix::mm::MprotectFlags;
use std::alloc::{self, Layout};
use std::sync::Arc;

use crate::nodes::{
    Expr, Func, FuncIdx, FuncType, Global, Instr, LocalIdx, MemType,
    Module as ParsedModule, NumType, RefType, TableType, TypeIdx, ValType, VecType,
};

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

#[derive(Clone, Copy, Debug)]
enum Value {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    V128(i128),
    RefNull,
    RefFunc(FuncAddr),
    RefExtern(ExternAddr),
}

enum FuncInstImpl {
    Guest(Arc<ModuleInstance>, TypeIdx, Func),
    Host(TKTK),
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
    layout: Layout,
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
        unsafe {
            rustix::mm::mprotect(bottom_of_stack.cast(), PAGE_SIZE, MprotectFlags::empty())
                .expect("failed to protect stack guard page");
        };

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
struct Module {
    parsed_module: ParsedModule,
}

#[derive(Debug)]
struct GuestFuncProto {
    func_type: FuncType,
    func: Func,
}

#[derive(Debug)]
struct HostFuncProto {
    func_type: FuncType,
    offset: u32,
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

struct Imports {
    globals: Vec<Value>,
}

impl Module {
    fn new(module: ParsedModule) -> Self {
        Self {
            parsed_module: module,
        }
    }

    fn instantiate(&self, imports: &Imports) -> anyhow::Result<ModuleInstance> {
        let _globals = self
            .parsed_module
            .global_section()
            .iter()
            .flat_map(|xs| xs.iter())
            .try_fold(Vec::new(), |mut globals, global| {
                let Global(global_type, Expr(instrs)) = global;

                if instrs.len() > 2 {
                    anyhow::bail!("multiple instr global initializers are not supported");
                }

                let global = match instrs.first() {
                    Some(Instr::I32Const(v)) => Value::I32(*v),
                    Some(Instr::I64Const(v)) => Value::I64(*v),
                    Some(Instr::F32Const(v)) => Value::F32(*v),
                    Some(Instr::F64Const(v)) => Value::F64(*v),
                    Some(Instr::GlobalGet(LocalIdx(idx))) => {
                        let idx = *idx as usize;
                        if idx < imports.globals.len() {
                            imports.globals[idx]
                        } else {
                            globals[idx]
                        }
                    }
                    Some(Instr::RefNull(_ref_type)) => Value::RefNull,
                    Some(Instr::RefFunc(FuncIdx(idx))) => Value::RefFunc(Addr(*idx)),
                    _ => anyhow::bail!("unsupported global initializer instruction"),
                };
                if !matches!(
                    (global_type.0, &global),
                    (ValType::NumType(NumType::I32), Value::I32(_))
                        | (ValType::NumType(NumType::I64), Value::I64(_))
                        | (ValType::NumType(NumType::I32), Value::F32(_))
                        | (ValType::NumType(NumType::F64), Value::F64(_))
                        | (ValType::VecType(VecType::V128), Value::V128(_))
                        | (ValType::RefType(RefType::FuncRef), Value::RefFunc(_))
                        | (ValType::RefType(RefType::FuncRef), Value::RefNull)
                        | (ValType::RefType(RefType::ExternRef), Value::RefExtern(_))
                ) {
                    anyhow::bail!("global type does not accept this value");
                }
                globals.push(global);
                Ok(globals)
            })?;

        if let Some(data) = self.parsed_module.data_section() {
            for _data_segment in data {
                todo!();
            }
        }

        if let Some(_types) = self.parsed_module.type_section() {}

        todo!();
    }

    /*
        public Instance instantiate(HostImports hostImports) {

            var dataSegments = new DataSegment[0];
            if (module.dataSection() != null) {
                dataSegments = module.dataSection().dataSegments();
            }

            var types = new FunctionType[0];
            // TODO i guess we should explode if this is the case, is this possible?
            if (module.typeSection() != null) {
                types = module.typeSection().types();
            }

            var numFuncTypes = 0;
            var funcSection = module.functionSection();
            if (funcSection != null) {
                numFuncTypes = funcSection.functionCount();
            }
            if (module.importSection() != null) {
                numFuncTypes +=
                        module.importSection().stream()
                                .filter(is -> is.importType() == ExternalType.FUNCTION)
                                .count();
            }

            FunctionBody[] functions = new FunctionBody[0];
            var codeSection = module.codeSection();
            if (codeSection != null) {
                functions = module.codeSection().functionBodies();
            }

            int importId = 0;
            var functionTypes = new int[numFuncTypes];
            var imports = new Import[0];
            var funcIdx = 0;

            if (module.importSection() != null) {
                int cnt = module.importSection().importCount();
                imports = new Import[cnt];
                for (int i = 0; i < cnt; i++) {
                    Import imprt = module.importSection().getImport(i);
                    switch (imprt.importType()) {
                        case FUNCTION:
                            {
                                var type = ((FunctionImport) imprt).typeIndex();
                                functionTypes[funcIdx] = type;
                                // The global function id increases on this table
                                // function ids are assigned on imports first
                                imports[importId++] = imprt;
                                funcIdx++;
                                break;
                            }
                        default:
                            imports[importId++] = imprt;
                            break;
                    }
                }
            }

            var mappedHostImports = mapHostImports(imports, hostImports);

            if (module.startSection() != null) {
                var export =
                        new Export(
                                START_FUNCTION_NAME,
                                (int) module.startSection().startIndex(),
                                ExternalType.FUNCTION);
                exports.put(START_FUNCTION_NAME, export);
            }

            if (module.functionSection() != null) {
                int cnt = module.functionSection().functionCount();
                for (int i = 0; i < cnt; i++) {
                    functionTypes[funcIdx++] = module.functionSection().getFunctionType(i);
                }
            }

            Table[] tables = new Table[0];
            if (module.tableSection() != null) {
                var tableLength = module.tableSection().tableCount();
                tables = new Table[tableLength];
                for (int i = 0; i < tableLength; i++) {
                    tables[i] = module.tableSection().getTable(i);
                }
            }

            Element[] elements = new Element[0];
            if (module.elementSection() != null) {
                elements = module.elementSection().elements();
            }

            Memory memory = null;
            if (module.memorySection() != null) {
                assert (mappedHostImports.memoryCount() == 0);

                var memories = module.memorySection();
                if (memories.memoryCount() > 1) {
                    throw new ChicoryException("Multiple memories are not supported");
                }
                if (memories.memoryCount() > 0) {
                    memory = new Memory(memories.getMemory(0).memoryLimits());
                }
            } else {
                if (mappedHostImports.memoryCount() > 0) {
                    assert (mappedHostImports.memoryCount() == 1);
                    if (mappedHostImports.memory(0) == null
                            || mappedHostImports.memory(0).memory() == null) {
                        throw new ChicoryException(
                                "Imported memory not defined, cannot run the program");
                    }
                    memory = mappedHostImports.memory(0).memory();
                } else {
                    // No memory defined
                }
            }

            var globalImportsOffset = 0;
            var functionImportsOffset = 0;
            var tablesImportsOffset = 0;
            for (int i = 0; i < imports.length; i++) {
                switch (imports[i].importType()) {
                    case GLOBAL:
                        globalImportsOffset++;
                        break;
                    case FUNCTION:
                        functionImportsOffset++;
                        break;
                    case TABLE:
                        tablesImportsOffset++;
                        break;
                    default:
                        break;
                }
            }

            return new Instance(
                    this,
                    globalInitializers,
                    globalImportsOffset,
                    functionImportsOffset,
                    tablesImportsOffset,
                    memory,
                    dataSegments,
                    functions,
                    types,
                    functionTypes,
                    mappedHostImports,
                    tables,
                    elements);
        }

        private HostImports mapHostImports(Import[] imports, HostImports hostImports) {
            int hostFuncNum = 0;
            int hostGlobalNum = 0;
            int hostMemNum = 0;
            int hostTableNum = 0;
            for (var imprt : imports) {
                switch (imprt.importType()) {
                    case FUNCTION:
                        hostFuncNum++;
                        break;
                    case GLOBAL:
                        hostGlobalNum++;
                        break;
                    case MEMORY:
                        hostMemNum++;
                        break;
                    case TABLE:
                        hostTableNum++;
                        break;
                }
            }

            // TODO: this can probably be refactored ...
            var hostFuncs = new HostFunction[hostFuncNum];
            var hostFuncIdx = 0;
            var hostGlobals = new HostGlobal[hostGlobalNum];
            var hostGlobalIdx = 0;
            var hostMems = new HostMemory[hostMemNum];
            var hostMemIdx = 0;
            var hostTables = new HostTable[hostTableNum];
            var hostTableIdx = 0;
            var hostIndex = new FromHost[hostFuncNum + hostGlobalNum + hostMemNum + hostTableNum];
            int cnt;
            for (var impIdx = 0; impIdx < imports.length; impIdx++) {
                var i = imports[impIdx];
                var name = i.moduleName() + "." + i.name();
                var found = false;
                switch (i.importType()) {
                    case FUNCTION:
                        cnt = hostImports.functionCount();
                        for (int j = 0; j < cnt; j++) {
                            HostFunction f = hostImports.function(j);
                            if (i.moduleName().equals(f.moduleName())
                                    && i.name().equals(f.fieldName())) {
                                hostFuncs[hostFuncIdx] = f;
                                hostIndex[impIdx] = f;
                                found = true;
                                break;
                            }
                        }
                        hostFuncIdx++;
                        break;
                    case GLOBAL:
                        cnt = hostImports.globalCount();
                        for (int j = 0; j < cnt; j++) {
                            HostGlobal g = hostImports.global(j);
                            if (i.moduleName().equals(g.moduleName())
                                    && i.name().equals(g.fieldName())) {
                                hostGlobals[hostGlobalIdx] = g;
                                hostIndex[impIdx] = g;
                                found = true;
                                break;
                            }
                        }
                        hostGlobalIdx++;
                        break;
                    case MEMORY:
                        cnt = hostImports.memoryCount();
                        for (int j = 0; j < cnt; j++) {
                            HostMemory m = hostImports.memory(j);
                            if (i.moduleName().equals(m.moduleName())
                                    && i.name().equals(m.fieldName())) {
                                hostMems[hostMemIdx] = m;
                                hostIndex[impIdx] = m;
                                found = true;
                                break;
                            }
                        }
                        hostMemIdx++;
                        break;
                    case TABLE:
                        cnt = hostImports.tableCount();
                        for (int j = 0; j < cnt; j++) {
                            HostTable t = hostImports.table(j);
                            if (i.moduleName().equals(t.moduleName())
                                    && i.name().equals(t.fieldName())) {
                                hostTables[hostTableIdx] = t;
                                hostIndex[impIdx] = t;
                                found = true;
                                break;
                            }
                        }
                        hostTableIdx++;
                        break;
                }
                if (!found) {
                    this.logger.warnf(
                            "Could not find host function for import number: %d named %s",
                            impIdx, name);
                }
            }

            var result = new HostImports(hostFuncs, hostGlobals, hostMems, hostTables);
            result.setIndex(hostIndex);
            return result;
        }
    */
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
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

    #[test]
    fn test_create_store() {
        let bytes = include_bytes!("../example.wasm");

        let _wasm = parse(bytes).unwrap();

        // let xs = Module::new(&wasm).unwrap();

        // dbg!(xs);
    }
}
