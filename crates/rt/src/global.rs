use uuasm_ir::{GlobalIdx, GlobalType, Import, Instr, Mutability, ValType};

use super::{
    imports::{Extern, GuestIndex, LookupImport},
    machine::MachineGlobalIndex,
};

#[derive(Debug, Clone)]
pub(super) enum GlobalInstImpl {
    Local(MachineGlobalIndex, Box<[Instr]>),
    Remote(GuestIndex, GlobalIdx),
}

#[derive(Debug, Clone)]
pub(crate) struct GlobalInst {
    r#type: GlobalType,
    pub(super) r#impl: GlobalInstImpl,
}

impl GlobalInst {
    pub(crate) fn resolve(
        ty: GlobalType,
        import: &Import,
        imports: &impl LookupImport,
    ) -> anyhow::Result<Self> {
        let Some(ext) = imports.lookup(import) else {
            anyhow::bail!("could not resolve {}/{}", import.module(), import.name());
        };

        let Extern::Global(module_idx, global_idx) = ext else {
            anyhow::bail!(
                "expected {}/{} to resolve to a global",
                import.module(),
                import.name()
            );
        };

        // TODO: validate type.
        Ok(Self {
            r#type: ty,
            r#impl: GlobalInstImpl::Remote(module_idx, global_idx),
        })
    }

    pub(super) fn initdata(&self) -> Option<(MachineGlobalIndex, &[Instr])> {
        match &self.r#impl {
            GlobalInstImpl::Local(idx, instrs) => Some((*idx, instrs.as_ref())),
            GlobalInstImpl::Remote(_, _) => None,
        }
    }

    pub(crate) fn typedef(&self) -> &GlobalType {
        &self.r#type
    }

    pub(crate) fn valtype(&self) -> ValType {
        self.r#type.0
    }

    pub(crate) fn mutability(&self) -> Mutability {
        self.r#type.1
    }

    pub(crate) fn new(ty: GlobalType, idx: MachineGlobalIndex, initializer: Box<[Instr]>) -> Self {
        Self {
            r#type: ty,
            r#impl: GlobalInstImpl::Local(idx, initializer),
        }
    }
}

/*
if instrs.len() > 2 {
    anyhow::bail!("multiple instr global initializers are not supported");
}


// TKTK: should we include this in "initializers" alongside active data/elem
// elements?
let global = match instrs.first() {
    Some(Instr::I32Const(v)) => Value::I32(*v),
    Some(Instr::I64Const(v)) => Value::I64(*v),
    Some(Instr::F32Const(v)) => Value::F32(*v),
    Some(Instr::F64Const(v)) => Value::F64(*v),
    Some(Instr::GlobalGet(GlobalIdx(idx))) => {
        let idx = *idx as usize;
        // globals[idx].value()
        todo!("lookup global from builder");
    }
    Some(Instr::RefNull(_ref_type)) => Value::RefNull,
    Some(Instr::RefFunc(FuncIdx(idx))) => Value::RefFunc(FuncIdx(*idx)),
    _ => anyhow::bail!("unsupported global initializer instruction"),
};
*/
