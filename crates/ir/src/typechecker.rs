#![allow(dead_code)]

use std::collections::LinkedList;
use thiserror::Error;

use crate::ValType;

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

enum BlockKind {
    Block,
    Loop,
}

struct CtrlFrame {
    kind: BlockKind,
    start_types: Vec<ValType>,
    end_types: Vec<ValType>,
    height: usize,
    unreachable: bool,
}

pub(crate) struct TypeChecker {
    vals: Vec<Val>,
    ctrls: LinkedList<CtrlFrame>,
}

#[derive(Clone, Debug, Error)]
pub enum TypeError {
    #[error("type stack not empty: {0} values left on stack")]
    UnexpectedTypes(usize),
    #[error("type mismatch: expected {expected}, got {received}")]
    TypeMismatch { received: Val, expected: Val },
    #[error("type stack empty")]
    StackUnderflow,
    #[error("block stack empty")]
    BlockUnderflow,
}

impl TypeChecker {
    fn push_val(&mut self, val: impl Into<Val>) {
        self.vals.push(val.into());
    }

    fn pop_val(&mut self, expect: Option<Val>) -> Result<Val, TypeError> {
        let Some(head) = self.ctrls.front() else {
            return Err(TypeError::BlockUnderflow);
        };

        let overflow = self.vals.len() == head.height;
        if overflow {
            return if head.unreachable {
                Ok(Val::Unknown)
            } else {
                Err(TypeError::StackUnderflow)
            };
        }

        let val = self.vals.pop().ok_or(TypeError::StackUnderflow)?;

        if let Some(expected) = expect {
            if expected != val {
                return Err(TypeError::TypeMismatch {
                    received: val,
                    expected,
                });
            }
        }
        Ok(val)
    }

    fn push_vals<T: Into<Val>, I: Iterator<Item = T>>(&mut self, vals: I) {
        for val in vals {
            self.push_val(val)
        }
    }

    fn pop_vals(&mut self, expect: impl Iterator<Item = Val>) -> Result<Vec<Val>, TypeError> {
        let mut values = expect
            .map(|expect| self.pop_val(Some(expect)))
            .collect::<Result<Vec<_>, _>>()?;
        values.reverse();
        Ok(values)
    }

    fn push_ctrl(
        &mut self,
        kind: BlockKind,
        input_types: Vec<ValType>,
        output_types: Vec<ValType>,
    ) {
        self.ctrls.push_front(CtrlFrame {
            kind,
            start_types: input_types.clone(),
            end_types: output_types,
            height: self.vals.len(),
            unreachable: false,
        });
        self.push_vals(input_types.iter().copied());
    }

    fn pop_ctrl(&mut self) -> Result<CtrlFrame, TypeError> {
        let Some(frame) = self.ctrls.pop_front() else {
            return Err(TypeError::BlockUnderflow);
        };
        self.pop_vals(frame.end_types.iter().copied().map(Into::<Val>::into))?;
        if self.vals.len() != frame.height {
            return Err(TypeError::UnexpectedTypes(self.vals.len() - frame.height));
        }
        Ok(frame)
    }

    fn label_types<'a>(&'_ mut self, frame: &'a CtrlFrame) -> &'a [ValType] {
        match frame.kind {
            BlockKind::Block => &frame.end_types,
            BlockKind::Loop => &frame.start_types,
        }
    }

    fn unreachable(&mut self) -> Result<(), TypeError> {
        let Some(frame) = self.ctrls.front_mut() else {
            return Err(TypeError::BlockUnderflow);
        };
        self.vals.truncate(frame.height);
        frame.unreachable = true;
        Ok(())
    }
}
