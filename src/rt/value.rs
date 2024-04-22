use crate::nodes::FuncIdx;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Value {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    V128(i128),
    RefNull,
    RefFunc(FuncIdx),
    RefExtern(u32),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::I32(l0), Self::I32(r0)) => l0 == r0,
            (Self::I64(l0), Self::I64(r0)) => l0 == r0,
            (Self::F32(l0), Self::F32(r0)) => l0 == r0,
            (Self::F64(l0), Self::F64(r0)) => l0 == r0,
            (Self::V128(l0), Self::V128(r0)) => l0 == r0,
            (Self::RefFunc(l0), Self::RefFunc(r0)) => l0 == r0,
            (Self::RefExtern(l0), Self::RefExtern(r0)) => l0 == r0,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl Value {
    #[cfg(test)]
    pub(crate) fn bit_eq(&self, rhs: &Self) -> anyhow::Result<()> {
        if match (self, rhs) {
            (Self::F32(l0), Self::F32(r0)) => unsafe {
                std::mem::transmute::<&f32, &u32>(l0) == std::mem::transmute::<&f32, &u32>(r0)
            },
            (Self::F64(l0), Self::F64(r0)) => unsafe {
                std::mem::transmute::<&f64, &u64>(l0) == std::mem::transmute::<&f64, &u64>(r0)
            },
            (lhs, rhs) => lhs == rhs,
        } {
            Ok(())
        } else {
            anyhow::bail!("{self:?} != {rhs:?}");
        }
    }

    pub(crate) fn as_usize(&self) -> Option<usize> {
        Some(match self {
            Value::I32(xs) => *xs as usize,
            Value::I64(xs) => *xs as usize,
            _ => return None,
        })
    }

    // This is distinct from "as_usize": i64's are not valid candidates
    // for memory offsets until mem64 lands in wasm.
    pub(crate) fn as_mem_offset(&self) -> anyhow::Result<usize> {
        self.as_i32().map(|xs| xs as usize)
    }

    pub(crate) fn as_i32(&self) -> anyhow::Result<i32> {
        Ok(match self {
            Value::I32(xs) => *xs,
            _ => anyhow::bail!("expected i32 value"),
        })
    }

    pub(crate) fn is_zero(&self) -> anyhow::Result<bool> {
        Ok(match self {
            Value::I32(xs) => *xs == 0,
            Value::I64(xs) => *xs == 0,
            Value::F32(xs) => *xs == 0.0,
            Value::F64(xs) => *xs == 0.0,
            _ => anyhow::bail!("invalid conversion of value to int via is_zero"),
        })
    }
}
