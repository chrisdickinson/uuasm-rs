#![allow(dead_code)]

use std::collections::LinkedList;
use thiserror::Error;

use crate::{
    ElemIdx, FuncIdx, GlobalIdx, GlobalType, Instr, LabelIdx, Local, LocalIdx, MemArg, Mutability,
    NumType, RefType, ResultType, TableIdx, TableType, Type, TypeIdx, ValType,
};

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Val {
    Unknown,
    Typed(ValType),
}

impl std::fmt::Display for Val {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown => f.write_str("unknown"),
            Self::Typed(t) => t.fmt(f),
        }
    }
}

impl Val {
    fn is_num(&self) -> bool {
        matches!(self, Self::Unknown | Self::Typed(ValType::NumType(_)),)
    }

    fn is_vec(&self) -> bool {
        matches!(self, Self::Unknown | Self::Typed(ValType::VecType(_)),)
    }

    fn is_ref(&self) -> bool {
        matches!(self, Self::Unknown | Self::Typed(ValType::RefType(_)),)
    }
}

impl From<ValType> for Val {
    fn from(value: ValType) -> Self {
        Self::Typed(value)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum BlockKind {
    Func,
    Block,
    Loop,
    If,
    Else,
    ConstantExpression,
}

#[derive(Clone, Debug)]
pub(crate) struct CtrlFrame {
    kind: BlockKind,
    pub(crate) start_types: Box<[ValType]>,
    pub(crate) end_types: Box<[ValType]>,
    height: usize,
    unreachable: bool,
}

#[derive(Default, Debug, Clone)]
pub(crate) struct TypeChecker {
    vals: Vec<Val>,
    ctrls: LinkedList<CtrlFrame>,
}

#[derive(Clone, Debug, Error)]
pub enum TypeError {
    #[error("type mismatch: stack not empty ({0} values left on stack)")]
    UnexpectedTypes(usize),

    #[error("type mismatch: expected {expected}, got {received}")]
    TypeMismatch { received: Val, expected: Val },

    #[error("table type mismatch; table refers to externref values")]
    TableTypeMismatch,

    #[error("select: type mismatch: invalid result arity")]
    SelectInvalidArity,

    #[error("type mismatch: br.table arity not uniform (expected {0}; got {1})")]
    BrTableArityMismatch(usize, usize),

    #[error("alignment must not be larger than natural (got {0}; must be < {1})")]
    InvalidLoadAlignment(u32, u32),

    #[error("global is immutable (global index={0})")]
    AssignmentToImmutableGlobal(u32),

    #[error("unknown label: out of range (got {0}; max is {1})")]
    InvalidLabelIndex(u32, u32),

    #[error("type mismatch: unexpected type (got {received})")]
    TypeclassMismatch { received: Val },

    #[error("type mismatch: select requires two numbers or two vectors")]
    InvalidSelection,

    #[error("constant expression required (got {0:?})")]
    ConstantExprRequired(Instr),

    #[error("type mismatch: stack empty")]
    StackUnderflow,

    #[error("block stack empty")]
    BlockUnderflow,

    #[error(
        "unknown global (module-local global references cannot be used in constant expressions)"
    )]
    LocalGlobalInConstantExpression,

    #[error(
        "type mismatch (alternate required when 'if' input length is different from result length)"
    )]
    AlternateRequired,
}

const I32: Val = Val::Typed(ValType::NumType(NumType::I32));
const FUNCREF: Val = Val::Typed(ValType::RefType(RefType::FuncRef));

macro_rules! conv {
    ($self:ident, () -> ()) => {};

    (@in F64) => { Val::Typed(ValType::NumType(NumType::F64)) };
    (@in F32) => { Val::Typed(ValType::NumType(NumType::F32)) };
    (@in I64) => { Val::Typed(ValType::NumType(NumType::I64)) };
    (@in I32) => { Val::Typed(ValType::NumType(NumType::I32)) };
    (@in V128) => { Val::Typed(ValType::VecType(VecType::V128)) };

    ($self:ident, () -> ($($result:ident),* $(,)?)) => {{
        $self.push_vals([
            $(conv!(@in $result),)*
        ].iter().copied());
    }};

    ($self:ident, ($($param:ident),* $(,)?) -> ()) => {{
        $self.pop_vals(&[
            $(conv!(@in $param),)*
        ])?;
    }};

    ($self:ident, ($($param:ident),* $(,)?) -> ($($result:ident),* $(,)?)) => {{
        $self.pop_vals(&[
            $(conv!(@in $param),)*
        ])?;
        $self.push_vals([
            $(conv!(@in $result),)*
        ].iter().copied());
    }};
}

impl TypeChecker {
    pub(crate) fn clear(&mut self) {
        self.vals.clear();
        self.ctrls.clear();
    }

    pub(crate) fn push_val(&mut self, val: impl Into<Val>) {
        let val = val.into();
        self.vals.push(val);
        #[cfg(any())]
        eprintln!("push {:?} <- {:?}", &self.vals, val);
    }

    pub(crate) fn pop_val(&mut self, expect: Option<Val>) -> Result<Val, TypeError> {
        let Some(head) = self.ctrls.front() else {
            return Err(TypeError::BlockUnderflow);
        };

        let overflow = self.vals.len() == head.height;
        if overflow {
            return if head.unreachable {
                #[cfg(any())]
                eprintln!("pop <unk>");
                Ok(Val::Unknown)
            } else {
                Err(TypeError::StackUnderflow)
            };
        }

        let actual = self.vals.pop().ok_or(TypeError::StackUnderflow)?;

        #[cfg(any())]
        eprintln!("pop {:?} -> {:?}", &self.vals, &actual);
        if let Some(expected) = expect {
            if expected != actual && expected != Val::Unknown && actual != Val::Unknown {
                return Err(TypeError::TypeMismatch {
                    received: actual,
                    expected,
                });
            }
        }
        Ok(actual)
    }

    pub(crate) fn push_vals<T: Into<Val>, I: Iterator<Item = T>>(&mut self, vals: I) {
        for val in vals {
            self.push_val(val)
        }
    }

    pub(crate) fn pop_vals<T: Copy + Into<Val>>(
        &mut self,
        expect: &[T],
    ) -> Result<Box<[Val]>, TypeError> {
        let mut into = LinkedList::new();
        for expectation in expect.iter().rev().copied() {
            let expectation: Val = expectation.into();
            into.push_front(self.pop_val(Some(expectation))?);
        }
        Ok(into.into_iter().collect())
    }

    pub(crate) fn push_ctrl(
        &mut self,
        kind: BlockKind,
        input_types: Box<[ValType]>,
        output_types: Box<[ValType]>,
    ) {
        let height = self.vals.len();
        self.push_vals(input_types.iter().copied());
        self.ctrls.push_front(CtrlFrame {
            kind,
            start_types: input_types,
            end_types: output_types,
            height,
            unreachable: false,
        });
    }

    pub(crate) fn pop_ctrl(&mut self) -> Result<CtrlFrame, TypeError> {
        let Some(frame) = self.ctrls.front() else {
            return Err(TypeError::BlockUnderflow);
        };
        let height = frame.height;
        let vals = frame.end_types.clone();
        let _ = frame;
        self.pop_vals(&vals)?;

        if self.vals.len() != height {
            return Err(TypeError::UnexpectedTypes(self.vals.len() - height));
        }
        let frame = self.ctrls.pop_front().unwrap();
        Ok(frame)
    }

    pub(crate) fn label_types(&self, frame: &CtrlFrame) -> Box<[ValType]> {
        match frame.kind {
            BlockKind::Loop => &frame.start_types,
            _ => &frame.end_types,
        }
        .iter()
        .copied()
        .collect()
    }

    pub(crate) fn unreachable(&mut self) -> Result<(), TypeError> {
        let Some(frame) = self.ctrls.front_mut() else {
            return Err(TypeError::BlockUnderflow);
        };
        self.vals.truncate(frame.height);
        frame.unreachable = true;
        Ok(())
    }

    pub(crate) fn trace(
        &mut self,
        instr: Instr,
        func_types: &[TypeIdx],
        types: &[Type],
        locals: &[Local],
        global_types: &[GlobalType],
        table_types: &[TableType],
        elem_types: &[RefType],
        global_import_boundary: u32,
    ) -> Result<Instr, TypeError> {
        let block_kind = self.ctrls.front().map(|ctrl| ctrl.kind).unwrap();
        if matches!(block_kind, BlockKind::ConstantExpression)
            && !matches!(
                &instr,
                Instr::I32Const(_)
                    | Instr::I64Const(_)
                    | Instr::F32Const(_)
                    | Instr::F64Const(_)
                    | Instr::GlobalGet(_)
                    | Instr::RefNull(_)
                    | Instr::RefFunc(_)
            )
        {
            return Err(TypeError::ConstantExprRequired(instr));
        }

        #[cfg(any())]
        dbg!(&instr);
        match &instr {
            Instr::CallIntrinsic(TypeIdx(idx), _) => {
                let type_idx = *idx as usize;
                let Type(ResultType(params), ResultType(results)) = &types[type_idx];

                self.pop_vals(params)?;
                self.push_vals(results.iter().copied());
            }

            Instr::TableInit(ElemIdx(elem_idx), TableIdx(table_idx)) => {
                let elem_type = &elem_types[*elem_idx as usize];
                let TableType(table_type, _) = &table_types[*table_idx as usize];

                if table_type != elem_type {
                    return Err(TypeError::TypeMismatch {
                        expected: Val::Typed(ValType::RefType(*table_type)),
                        received: Val::Typed(ValType::RefType(*elem_type)),
                    });
                }

                conv!(self, (I32, I32, I32) -> ())
            }
            Instr::ElemDrop(_) => {}
            Instr::TableCopy(TableIdx(src), TableIdx(dest)) => {
                let TableType(src_type, _) = &table_types[*src as usize];
                let TableType(dest_type, _) = &table_types[*dest as usize];
                if *src_type != *dest_type {
                    return Err(TypeError::TypeMismatch {
                        expected: Val::Typed(ValType::RefType(*src_type)),
                        received: Val::Typed(ValType::RefType(*dest_type)),
                    });
                }
                conv!(self, (I32, I32, I32) -> ())
            }

            Instr::TableGrow(TableIdx(table_idx)) => {
                let TableType(table_type, _) = table_types[*table_idx as usize];
                self.pop_vals(&[Val::Typed(ValType::RefType(table_type)), I32])?;
                self.push_val(I32);
            }

            Instr::TableSize(_) => {
                self.push_val(I32);
            }

            Instr::TableFill(TableIdx(table_idx)) => {
                let TableType(table_type, _) = table_types[*table_idx as usize];
                self.pop_vals(&[I32, Val::Typed(ValType::RefType(table_type)), I32])?;
            }

            Instr::Unreachable => {
                self.unreachable()?;
            }
            Instr::Nop => {}
            Instr::Block(_, _) | Instr::Loop(_, _) | Instr::IfElse(_, _, _) => {
                let frame = self.pop_ctrl()?;
                self.push_vals(frame.end_types.iter().copied());
            }

            Instr::If(_, _) => {
                let frame = self.pop_ctrl()?;
                if frame.start_types.len() != frame.end_types.len() {
                    return Err(TypeError::AlternateRequired);
                }
                self.push_vals(frame.end_types.iter().copied());
            }

            Instr::Br(LabelIdx(idx)) => {
                let label_idx = *idx as usize;
                let Some(frame) = self.ctrls.iter().nth(label_idx) else {
                    return Err(TypeError::InvalidLabelIndex(*idx, self.ctrls.len() as u32));
                };

                let vals = self.label_types(frame);
                self.pop_vals(&vals)?;
                self.unreachable()?;
            }
            Instr::BrIf(LabelIdx(idx)) => {
                self.pop_val(Some(I32))?;
                let label_idx = *idx as usize;
                let Some(frame) = self.ctrls.iter().nth(label_idx) else {
                    return Err(TypeError::InvalidLabelIndex(*idx, self.ctrls.len() as u32));
                };

                // XXX: be careful here: we specifically want to replace the values on the stack
                // with the **label_types values**. (iow: br_if can launder Val::Unknown values
                // into Val::Typed values.)
                let label_vals = self.label_types(frame);
                self.pop_vals(&label_vals)?;
                self.push_vals(label_vals.iter().copied());
            }

            Instr::BrTable(labels, LabelIdx(idx)) => {
                self.pop_val(Some(I32))?;
                let label_idx = *idx as usize;
                let Some(frame) = self.ctrls.iter().nth(label_idx) else {
                    return Err(TypeError::InvalidLabelIndex(*idx, self.ctrls.len() as u32));
                };
                let vals = self.label_types(frame);

                let arity = vals.len();
                let _ = frame;
                for LabelIdx(idx) in labels {
                    let label_idx = *idx as usize;
                    let Some(frame) = self.ctrls.iter().nth(label_idx) else {
                        return Err(TypeError::InvalidLabelIndex(*idx, self.ctrls.len() as u32));
                    };
                    let vals: Box<[Val]> = self
                        .label_types(frame)
                        .iter()
                        .copied()
                        .map(Into::into)
                        .collect();

                    if vals.len() != arity {
                        return Err(TypeError::BrTableArityMismatch(arity, vals.len()));
                    }

                    let vals = self.pop_vals(&vals)?;
                    self.push_vals(vals.iter().copied());
                }
                self.pop_vals(&vals)?;
                self.unreachable()?;
            }

            Instr::Drop => {
                self.pop_val(None)?;
            }

            Instr::Return => {
                let Some(frame) = self.ctrls.iter().last() else {
                    // TODO: better error for "we tried to return outside the context of a control
                    // stack"
                    return Err(TypeError::InvalidLabelIndex(0, self.ctrls.len() as u32));
                };

                let vals: Box<[Val]> = self
                    .label_types(frame)
                    .iter()
                    .copied()
                    .map(Into::into)
                    .collect();
                self.pop_vals(&vals)?;
                self.unreachable()?;
            }

            Instr::Call(FuncIdx(idx)) => {
                let func_idx = *idx as usize;
                let TypeIdx(type_idx) = &func_types[func_idx];
                let Type(ResultType(params), ResultType(results)) = &types[*type_idx as usize];

                self.pop_vals(params)?;
                self.push_vals(results.iter().copied());
            }

            Instr::CallIndirect(TypeIdx(type_idx), TableIdx(table_idx)) => {
                let TableType(table_type, _) = table_types[*table_idx as usize];
                if table_type != RefType::FuncRef {
                    return Err(TypeError::TableTypeMismatch);
                }

                // TODO: use table_idx to check that the table is funcref
                self.pop_val(Some(I32))?;
                let Type(ResultType(params), ResultType(results)) = &types[*type_idx as usize];

                self.pop_vals(params)?;
                self.push_vals(results.iter().copied());
            }

            Instr::RefNull(ref_type) => {
                self.push_val(ValType::RefType(*ref_type));
            }

            Instr::RefIsNull => {
                let val = self.pop_val(None)?;
                if !val.is_ref() {
                    return Err(TypeError::TypeclassMismatch { received: val });
                }
                self.push_val(I32);
            }

            Instr::RefFunc(FuncIdx(_)) => self.push_val(FUNCREF),

            Instr::Select(results) => {
                if results.len() != 1 {
                    return Err(TypeError::SelectInvalidArity);
                }
                self.pop_val(Some(I32))?;
                self.pop_vals(results)?;
                self.pop_vals(results)?;
                self.push_vals(results.iter().copied());
            }
            Instr::SelectEmpty => {
                self.pop_val(Some(I32))?;
                let t1 = self.pop_val(None).map_err(|e| {
                    if matches!(e, TypeError::StackUnderflow) {
                        return TypeError::SelectInvalidArity;
                    }
                    e
                })?;
                let t2 = self.pop_val(None).map_err(|e| {
                    if matches!(e, TypeError::StackUnderflow) {
                        return TypeError::SelectInvalidArity;
                    }
                    e
                })?;

                if !((t1.is_num() && t2.is_num()) || (t1.is_vec() && t1.is_vec())) {
                    return Err(TypeError::InvalidSelection);
                }

                if t1 != t2 && t1 != Val::Unknown && t2 != Val::Unknown {
                    return Err(TypeError::InvalidSelection);
                }

                self.push_val(if t1 == Val::Unknown { t2 } else { t1 });

                let val_type = if let Val::Typed(v) = t1 {
                    v
                } else {
                    // assumption: if we have an unknown value, this instr can never
                    // be reached. So we fake an i32.
                    ValType::NumType(NumType::I32)
                };
                // rewrite the instruction to annotate the type
                return Ok(Instr::Select(Box::new([val_type])));
            }

            Instr::LocalGet(LocalIdx(local_idx)) => {
                let mut base = 0;
                let Local(_, local_type) = locals
                    .iter()
                    .find(|Local(count, _)| {
                        let found = base <= *local_idx && (base + count) > *local_idx;
                        base += count;
                        found
                    })
                    .unwrap();

                self.push_val(Val::Typed(*local_type));
            }

            Instr::LocalSet(LocalIdx(local_idx)) => {
                let mut base = 0;
                let Local(_, local_type) = locals
                    .iter()
                    .find(|Local(count, _)| {
                        let found = base <= *local_idx && (base + count) > *local_idx;
                        base += count;
                        found
                    })
                    .unwrap();
                self.pop_val(Some(Val::Typed(*local_type)))?;
            }
            Instr::LocalTee(LocalIdx(local_idx)) => {
                let mut base = 0;
                let Local(_, local_type) = locals
                    .iter()
                    .find(|Local(count, _)| {
                        let found = base <= *local_idx && (base + count) > *local_idx;
                        base += count;
                        found
                    })
                    .unwrap();
                let val = self.pop_val(Some(Val::Typed(*local_type)))?;
                self.push_val(val);
            }

            Instr::GlobalGet(GlobalIdx(global_idx)) => {
                let GlobalType(val_type, mutability) = global_types[*global_idx as usize];

                if matches!(block_kind, BlockKind::ConstantExpression) {
                    if matches!(mutability, Mutability::Variable) {
                        return Err(TypeError::ConstantExprRequired(instr));
                    }

                    if *global_idx >= global_import_boundary {
                        return Err(TypeError::LocalGlobalInConstantExpression);
                    }
                }
                self.push_val(val_type);
            }

            Instr::GlobalSet(GlobalIdx(global_idx)) => {
                let GlobalType(val_type, mutability) = global_types[*global_idx as usize];

                if mutability != Mutability::Variable {
                    return Err(TypeError::AssignmentToImmutableGlobal(*global_idx));
                }

                self.pop_val(Some(Val::Typed(val_type)))?;
            }

            Instr::TableGet(TableIdx(table_idx)) => {
                let TableType(table_type, _) = table_types[*table_idx as usize];
                self.pop_val(Some(I32))?;
                self.push_val(Val::Typed(ValType::RefType(table_type)));
            }

            Instr::TableSet(TableIdx(table_idx)) => {
                let TableType(table_type, _) = table_types[*table_idx as usize];
                self.pop_val(Some(Val::Typed(ValType::RefType(table_type))))?;
                self.pop_val(Some(I32))?;
            }

            Instr::I32Load(MemArg(align, _)) => {
                if (1 << *align) > 4 {
                    return Err(TypeError::InvalidLoadAlignment(*align, 4));
                }
                conv!(self, (I32) -> (I32))
            }
            Instr::I64Load(MemArg(align, _)) => {
                if (1 << *align) > 8 {
                    return Err(TypeError::InvalidLoadAlignment(*align, 8));
                }
                conv!(self, (I32) -> (I64))
            }
            Instr::F32Load(MemArg(align, _)) => {
                if (1 << *align) > 4 {
                    return Err(TypeError::InvalidLoadAlignment(*align, 4));
                }
                conv!(self, (I32) -> (F32))
            }
            Instr::F64Load(MemArg(align, _)) => {
                if (1 << *align) > 8 {
                    return Err(TypeError::InvalidLoadAlignment(*align, 8));
                }
                conv!(self, (I32) -> (F64))
            }

            Instr::I32Load8U(MemArg(align, _)) | Instr::I32Load8S(MemArg(align, _)) => {
                if (1 << *align) > 1 {
                    return Err(TypeError::InvalidLoadAlignment(*align, 1));
                }
                conv!(self, (I32) -> (I32))
            }

            Instr::I32Load16S(MemArg(align, _)) | Instr::I32Load16U(MemArg(align, _)) => {
                if (1 << *align) > 2 {
                    return Err(TypeError::InvalidLoadAlignment(*align, 2));
                }
                conv!(self, (I32) -> (I32))
            }

            Instr::I64Load8S(MemArg(align, _)) | Instr::I64Load8U(MemArg(align, _)) => {
                if (1 << *align) > 1 {
                    return Err(TypeError::InvalidLoadAlignment(*align, 1));
                }
                conv!(self, (I32) -> (I64))
            }

            Instr::I64Load16S(MemArg(align, _)) | Instr::I64Load16U(MemArg(align, _)) => {
                if (1 << *align) > 2 {
                    return Err(TypeError::InvalidLoadAlignment(*align, 2));
                }
                conv!(self, (I32) -> (I64))
            }

            Instr::I64Load32S(MemArg(align, _)) | Instr::I64Load32U(MemArg(align, _)) => {
                if (1 << *align) > 4 {
                    return Err(TypeError::InvalidLoadAlignment(*align, 4));
                }
                conv!(self, (I32) -> (I64))
            }

            Instr::I32Store(MemArg(align, _)) => {
                if (1 << *align) > 4 {
                    return Err(TypeError::InvalidLoadAlignment(*align, 4));
                }

                conv!(self, (I32, I32) -> ())
            }
            Instr::I64Store(MemArg(align, _)) => {
                if (1 << *align) > 8 {
                    return Err(TypeError::InvalidLoadAlignment(*align, 8));
                }

                conv!(self, (I32, I64) -> ())
            }
            Instr::F32Store(MemArg(align, _)) => {
                if (1 << *align) > 4 {
                    return Err(TypeError::InvalidLoadAlignment(*align, 4));
                }

                conv!(self, (I32, F32) -> ())
            }
            Instr::F64Store(MemArg(align, _)) => {
                if (1 << *align) > 8 {
                    return Err(TypeError::InvalidLoadAlignment(*align, 8));
                }

                conv!(self, (I32, F64) -> ())
            }
            Instr::I32Store8(MemArg(align, _)) => {
                if (1 << *align) > 1 {
                    return Err(TypeError::InvalidLoadAlignment(*align, 1));
                }
                conv!(self, (I32, I32) -> ())
            }

            Instr::I32Store16(MemArg(align, _)) => {
                if (1 << *align) > 2 {
                    return Err(TypeError::InvalidLoadAlignment(*align, 2));
                }
                conv!(self, (I32, I32) -> ())
            }

            Instr::I64Store8(MemArg(align, _)) => {
                if (1 << *align) > 1 {
                    return Err(TypeError::InvalidLoadAlignment(*align, 1));
                }
                conv!(self, (I32, I64) -> ())
            }

            Instr::I64Store16(MemArg(align, _)) => {
                if (1 << *align) > 2 {
                    return Err(TypeError::InvalidLoadAlignment(*align, 2));
                }
                conv!(self, (I32, I64) -> ())
            }

            Instr::I64Store32(MemArg(align, _)) => {
                if (1 << *align) > 4 {
                    return Err(TypeError::InvalidLoadAlignment(*align, 4));
                }
                conv!(self, (I32, I64) -> ())
            }
            Instr::MemorySize(_) => conv!(self, () -> (I32)),
            Instr::MemoryGrow(_) => conv!(self, (I32) -> (I32)),
            Instr::MemoryInit(_, _) => conv!(self, (I32, I32, I32) -> ()),
            Instr::DataDrop(_) => {}
            Instr::MemoryCopy(_, _) | Instr::MemoryFill(_) => conv!(self, (I32, I32, I32) -> ()),

            Instr::I32Const(_) => conv!(self, () -> (I32)),
            Instr::I64Const(_) => conv!(self, () -> (I64)),
            Instr::F32Const(_) => conv!(self, () -> (F32)),
            Instr::F64Const(_) => conv!(self, () -> (F64)),

            Instr::I32Eqz => conv!(self, (I32) -> (I32)),
            Instr::I32Eq
            | Instr::I32Ne
            | Instr::I32LtS
            | Instr::I32LtU
            | Instr::I32GtS
            | Instr::I32GtU
            | Instr::I32LeS
            | Instr::I32LeU
            | Instr::I32GeS
            | Instr::I32GeU => conv!(self, (I32, I32) -> (I32)),

            Instr::I64Eqz => conv!(self, (I64) -> (I32)),
            Instr::I64Eq
            | Instr::I64Ne
            | Instr::I64LtS
            | Instr::I64LtU
            | Instr::I64GtS
            | Instr::I64GtU
            | Instr::I64LeS
            | Instr::I64LeU
            | Instr::I64GeS
            | Instr::I64GeU => conv!(self, (I64, I64) -> (I32)),

            Instr::F32Eq
            | Instr::F32Ne
            | Instr::F32Lt
            | Instr::F32Gt
            | Instr::F32Le
            | Instr::F32Ge => conv!(self, (F32, F32) -> (I32)),

            Instr::F64Eq
            | Instr::F64Ne
            | Instr::F64Lt
            | Instr::F64Gt
            | Instr::F64Le
            | Instr::F64Ge => conv!(self, (F64, F64) -> (I32)),

            Instr::I32Clz | Instr::I32Ctz | Instr::I32Popcnt => conv!(self, (I32) -> (I32)),

            Instr::I32Add
            | Instr::I32Sub
            | Instr::I32Mul
            | Instr::I32DivS
            | Instr::I32DivU
            | Instr::I32RemS
            | Instr::I32RemU
            | Instr::I32And
            | Instr::I32Ior
            | Instr::I32Xor
            | Instr::I32Shl
            | Instr::I32ShrS
            | Instr::I32ShrU
            | Instr::I32Rol
            | Instr::I32Ror => conv!(self, (I32, I32) -> (I32)),

            Instr::I64Clz | Instr::I64Ctz | Instr::I64Popcnt => conv!(self, (I64) -> (I64)),

            Instr::I64Add
            | Instr::I64Sub
            | Instr::I64Mul
            | Instr::I64DivS
            | Instr::I64DivU
            | Instr::I64RemS
            | Instr::I64RemU
            | Instr::I64And
            | Instr::I64Ior
            | Instr::I64Xor
            | Instr::I64Shl
            | Instr::I64ShrS
            | Instr::I64ShrU
            | Instr::I64Rol
            | Instr::I64Ror => conv!(self, (I64, I64) -> (I64)),

            Instr::F32Abs
            | Instr::F32Neg
            | Instr::F32Ceil
            | Instr::F32Floor
            | Instr::F32Trunc
            | Instr::F32NearestInt
            | Instr::F32Sqrt => conv!(self, (F32) -> (F32)),

            Instr::F32Add
            | Instr::F32Sub
            | Instr::F32Mul
            | Instr::F32Div
            | Instr::F32Min
            | Instr::F32Max
            | Instr::F32CopySign => conv!(self, (F32, F32) -> (F32)),

            Instr::F64Abs
            | Instr::F64Neg
            | Instr::F64Ceil
            | Instr::F64Floor
            | Instr::F64Trunc
            | Instr::F64NearestInt
            | Instr::F64Sqrt => conv!(self, (F64) -> (F64)),

            Instr::F64Add
            | Instr::F64Sub
            | Instr::F64Mul
            | Instr::F64Div
            | Instr::F64Min
            | Instr::F64Max
            | Instr::F64CopySign => conv!(self, (F64, F64) -> (F64)),

            Instr::I32ConvertI64 => conv!(self, (I64) -> (I32)),
            Instr::I32SConvertF32 | Instr::I32UConvertF32 => conv!(self, (F32) -> (I32)),
            Instr::I32SConvertF64 | Instr::I32UConvertF64 => conv!(self, (F64) -> (I32)),
            Instr::I64SConvertI32 | Instr::I64UConvertI32 => conv!(self, (I32) -> (I64)),
            Instr::I64SConvertF32 | Instr::I64UConvertF32 => conv!(self, (F32) -> (I64)),
            Instr::I64SConvertF64 | Instr::I64UConvertF64 => conv!(self, (F64) -> (I64)),

            Instr::F32SConvertI32 | Instr::F32UConvertI32 => conv!(self, (I32) -> (F32)),
            Instr::F32SConvertI64 | Instr::F32UConvertI64 => conv!(self, (I64) -> (F32)),
            Instr::F32ConvertF64 => conv!(self, (F64) -> (F32)),
            Instr::F64SConvertI32 | Instr::F64UConvertI32 => conv!(self, (I32) -> (F64)),
            Instr::F64SConvertI64 | Instr::F64UConvertI64 => conv!(self, (I64) -> (F64)),
            Instr::F64ConvertF32 => conv!(self, (F32) -> (F64)),
            Instr::I32SConvertSatF32 | Instr::I32UConvertSatF32 => conv!(self, (F32) -> (I32)),
            Instr::I32SConvertSatF64 | Instr::I32UConvertSatF64 => conv!(self, (F64) -> (I32)),
            Instr::I64SConvertSatF32 | Instr::I64UConvertSatF32 => conv!(self, (F32) -> (I64)),
            Instr::I64SConvertSatF64 | Instr::I64UConvertSatF64 => conv!(self, (F64) -> (I64)),
            Instr::I32ReinterpretF32 => conv!(self, (F32) -> (I32)),
            Instr::I64ReinterpretF64 => conv!(self, (F64) -> (I64)),

            Instr::F32ReinterpretI32 => conv!(self, (I32) -> (F32)),

            Instr::F64ReinterpretI64 => conv!(self, (I64) -> (F64)),

            // unary accept I32 produce I32
            Instr::I32SExtendI8 | Instr::I32SExtendI16 => conv!(self, (I32) -> (I32)),

            // unary accept I64 produce I64
            Instr::I64SExtendI8 | Instr::I64SExtendI16 | Instr::I64SExtendI32 => {
                conv!(self, (I64) -> (I64))
            }
        }
        Ok(instr)
    }
}
