use crate::nodes::FuncIdx;

#[derive(Clone, Copy, Debug)]
pub enum Value {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    V128(i128),
    RefNull,
    RefFunc(FuncIdx),
    RefExtern(u32),
    #[cfg(test)]
    F32CanonicalNaN,
    #[cfg(test)]
    F64CanonicalNaN,
    #[cfg(test)]
    F32ArithmeticNaN,
    #[cfg(test)]
    F64ArithmeticNaN,
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
        let (lbits, rbits) = match (self, rhs) {
            (Self::F32(l0f), Self::F32(r0f)) => {
                (l0f.to_bits() as u64, r0f.to_bits() as u64)
            },
            (Self::F64(l0f), Self::F64(r0f)) => {
                (l0f.to_bits(), r0f.to_bits())
            },
            (Self::F32(l0f), Self::F32CanonicalNaN) => {
                let l0 = l0f.to_bits();
                if (l0 & 0x7fff_ffff) == 0x7fc0_0000 {
                    return Ok(())
                } else {
                    anyhow::bail!("F32({l0f}) ({l0:x}) not canonical nan");
                }
            },
            (Self::F64(l0f), Self::F64CanonicalNaN) => {
                let l0 = l0f.to_bits();
                if (l0 & 0x7fff_ffff_ffff_ffff) == 0x7ff8_0000_0000_0000 {
                    return Ok(())
                } else {
                    anyhow::bail!("F64({l0f}) ({l0:x}) not canonical nan");
                }
            },
            (Self::F32(l0f), Self::F32ArithmeticNaN) => {
                let l0 = l0f.to_bits();
                if (l0 & 0x7f80_0000) == 0x7f80_0000 && (l0 & 0x0040_0000) == 0x0040_0000 {
                    return Ok(())
                } else {
                    anyhow::bail!("F32({l0f}) ({l0:x}) not arithmetic nan");
                }
            },
            (Self::F64(l0f), Self::F64ArithmeticNaN) => {
                let l0 = l0f.to_bits();
                if (l0 & 0x7ff0_0000_0000_0000) == 0x7ff0_0000_0000_0000 && (l0 & 0x0008_0000_0000_0000) == 0x0008_0000_0000_0000 {
                    return Ok(())
                } else {
                    anyhow::bail!("F64({l0f}) ({l0:x}) not arithmetic nan");
                }
            },
            (Self::I32(l0), Self::I32(r0)) => (*l0 as u64, *r0 as u64),
            (Self::I64(l0), Self::I32(r0)) => (*l0 as u64, *r0 as u64),
            (Self::I32(l0), Self::I64(r0)) => (*l0 as u64, *r0 as u64),
            (Self::I64(l0), Self::I64(r0)) => (*l0 as u64, *r0 as u64),
            (lhs, rhs) => {
                if lhs == rhs {
                    return Ok(());
                } else {
                    anyhow::bail!("{self:?} != {rhs:?}");
                }
            }
        };

        if lbits != rbits {
            anyhow::bail!("{self:?} ({lbits:x}) != {rhs:?} ({rbits:x})");
        }
        Ok(())
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
