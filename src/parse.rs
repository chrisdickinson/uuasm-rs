use core::ops::Deref;
use nom::error::ParseError;
use nom_locate::LocatedSpan;
use std::fmt::Debug;

use crate::nodes::*;

type Span<'a> = LocatedSpan<&'a [u8]>;

macro_rules! impl_parse_for_newtype {
    ($type:ident, $innertype:ident) => {
        impl<'a> ParseWasmBinary<'a> for $type {
            fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
                input: Span<'a>,
            ) -> nom::IResult<Span<'a>, Self, E> {
                use nom::combinator::map;
                map($innertype::from_wasm_bytes, $type)(input)
            }
        }
    };
}

pub(crate) trait ParseWasmBinary<'a>: Sized {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        b: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E>;
}

impl<'a, T: Debug + ParseWasmBinary<'a>> ParseWasmBinary<'a> for Vec<T> {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        b: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        let (mut rest, sz) = <u32 as ParseWasmBinary>::from_wasm_bytes(b)?;
        let mut v = Vec::with_capacity(sz as usize);
        for _ in 0..sz {
            let (r, item) = <T as ParseWasmBinary>::from_wasm_bytes(rest)?;
            v.push(item);
            rest = r;
        }
        Ok((rest, v))
    }
}

impl<'a> ParseWasmBinary<'a> for ByteVec<'a> {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::bytes::complete::take;
        let (input, sz) = <u32 as ParseWasmBinary>::from_wasm_bytes(input)?;
        let (input, span) = take(sz as usize)(input)?;
        Ok((input, ByteVec(span.deref())))
    }
}

impl<'a> ParseWasmBinary<'a> for &'a str {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::bytes::complete::take;
        let (input, sz) = <u32 as ParseWasmBinary>::from_wasm_bytes(input)?;
        let (input, span) = take(sz as usize)(input)?;
        let Ok(xs) = std::str::from_utf8(&span) else {
            return nom::combinator::fail(input);
        };

        Ok((input, xs))
    }
}

impl<'a> ParseWasmBinary<'a> for Name<'a> {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        let (input, sz) = <&'a str>::from_wasm_bytes(input)?;

        Ok((input, Name(sz)))
    }
}

macro_rules! parse_leb128 {
    ($type:ident, signed) => {
        impl<'a> ParseWasmBinary<'a> for $type {
            fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
                input: Span<'a>,
            ) -> nom::IResult<Span<'a>, Self, E> {
                use nom::{
                    bytes::complete::{take, take_till},
                    combinator::fail,
                };

                let (input, leb_bytes) = take_till(|xs| xs & 0x80 == 0)(input)?;
                let (input, last) = take(1usize)(input)?;

                if leb_bytes.len() + 1 > divrem::DivCeil::div_ceil($type::BITS, 7) as usize {
                    return fail::<_, Self, _>(input);
                }

                let mut result: $type = 0;
                let mut shift = 0;
                for xs in &leb_bytes[..] {
                    result |= ((xs & 0x7f) as $type) << shift;
                    shift += 7;
                }
                result += ((last[0] & 0x7f) as $type) << shift;
                shift += 7;
                if shift < $type::BITS && (last[0] & 0x40) == 0x40 {
                    result |= !0 << shift;
                }

                Ok((input, result))
            }
        }
    };

    ($type:ident, unsigned) => {
        impl<'a> ParseWasmBinary<'a> for $type {
            fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
                input: Span<'a>,
            ) -> nom::IResult<Span<'a>, Self, E> {
                use nom::{
                    bytes::complete::{take, take_till},
                    combinator::fail,
                };

                let (input, leb_bytes) = take_till(|xs| xs & 0x80 == 0)(input)?;
                let (input, last) = take(1usize)(input)?;

                if leb_bytes.len() + 1 > divrem::DivCeil::div_ceil($type::BITS, 7) as usize {
                    return fail::<_, Self, _>(input);
                }

                let mut result: $type = 0;
                let mut shift = 0;
                for xs in &leb_bytes[..] {
                    result |= ((xs & 0x7f) as $type) << shift;
                    shift += 7;
                }
                result += ((last[0] & 0x7f) as $type) << shift;

                Ok((input, result))
            }
        }
    };
}

parse_leb128!(i32, signed);
parse_leb128!(i64, signed);
parse_leb128!(u32, unsigned);
parse_leb128!(u64, unsigned);

impl<'a> ParseWasmBinary<'a> for f32 {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        b: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::number::complete::le_f32;
        le_f32(b)
    }
}

impl<'a> ParseWasmBinary<'a> for f64 {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        b: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::number::complete::le_f64;
        le_f64(b)
    }
}

impl<'a> ParseWasmBinary<'a> for NumType {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::bytes::complete::take;

        let (input, byte) = take(1usize)(input)?;

        Ok((
            input,
            match byte[0] {
                0x7f => NumType::I32,
                0x7e => NumType::I64,
                0x7d => NumType::F32,
                0x7c => NumType::F64,
                _ => {
                    return Err(nom::Err::Error(nom::error::make_error(
                        input,
                        nom::error::ErrorKind::Alt,
                    )))
                }
            },
        ))
    }
}

impl<'a> ParseWasmBinary<'a> for VecType {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::bytes::complete::take;

        let (input, byte) = take(1usize)(input)?;

        Ok((
            input,
            match byte[0] {
                0x7b => VecType::V128,
                _ => {
                    return Err(nom::Err::Error(nom::error::make_error(
                        input,
                        nom::error::ErrorKind::Alt,
                    )))
                }
            },
        ))
    }
}

impl<'a> ParseWasmBinary<'a> for RefType {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::bytes::complete::take;

        let (input, byte) = take(1usize)(input)?;

        Ok((
            input,
            match byte[0] {
                0x70 => RefType::FuncRef,
                0x6f => RefType::ExternRef,
                _ => {
                    return Err(nom::Err::Error(nom::error::make_error(
                        input,
                        nom::error::ErrorKind::Alt,
                    )))
                }
            },
        ))
    }
}

impl<'a> ParseWasmBinary<'a> for ValType {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::{branch::alt, combinator::map};

        alt((
            map(NumType::from_wasm_bytes, ValType::NumType),
            map(VecType::from_wasm_bytes, ValType::VecType),
            map(RefType::from_wasm_bytes, ValType::RefType),
        ))(input)
    }
}

impl<'a> ParseWasmBinary<'a> for ResultType {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        b: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::combinator::map;
        map(Vec::<ValType>::from_wasm_bytes, ResultType)(b)
    }
}

impl<'a> ParseWasmBinary<'a> for Type {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        b: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::{
            bytes::complete::tag,
            combinator::map,
            sequence::{preceded, tuple},
        };
        map(
            preceded(
                tag([0x60]),
                tuple((ResultType::from_wasm_bytes, ResultType::from_wasm_bytes)),
            ),
            |(args, rets)| Type(args, rets),
        )(b)
    }
}

impl<'a> ParseWasmBinary<'a> for Limits {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        b: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::{
            branch::alt,
            bytes::complete::tag,
            combinator::map,
            sequence::{preceded, tuple},
        };

        alt((
            map(preceded(tag([0x00]), u32::from_wasm_bytes), Limits::Min),
            map(
                preceded(
                    tag([0x01]),
                    tuple((u32::from_wasm_bytes, u32::from_wasm_bytes)),
                ),
                |(min, max)| Limits::Range(min, max),
            ),
        ))(b)
    }
}

impl<'a> ParseWasmBinary<'a> for MemType {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        b: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::combinator::map;
        map(Limits::from_wasm_bytes, MemType)(b)
    }
}

impl<'a> ParseWasmBinary<'a> for Global {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::{combinator::map, sequence::tuple};
        map(
            tuple((GlobalType::from_wasm_bytes, Expr::from_wasm_bytes)),
            |(x, y)| Global(x, y),
        )(input)
    }
}

impl<'a> ParseWasmBinary<'a> for TableType {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        b: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::{combinator::map, sequence::tuple};
        map(
            tuple((RefType::from_wasm_bytes, Limits::from_wasm_bytes)),
            |(rt, limits)| TableType(rt, limits),
        )(b)
    }
}

impl<'a> ParseWasmBinary<'a> for Mutability {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::bytes::complete::take;

        let (input, byte) = take(1usize)(input)?;

        Ok((
            input,
            match byte[0] {
                0x00 => Mutability::Const,
                0x01 => Mutability::Variable,
                _ => {
                    return Err(nom::Err::Error(nom::error::make_error(
                        input,
                        nom::error::ErrorKind::Alt,
                    )))
                }
            },
        ))
    }
}

impl<'a> ParseWasmBinary<'a> for GlobalType {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        b: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::{combinator::map, sequence::tuple};
        map(
            tuple((ValType::from_wasm_bytes, Mutability::from_wasm_bytes)),
            |(ty, mutability)| GlobalType(ty, mutability),
        )(b)
    }
}

impl<'a> ParseWasmBinary<'a> for BlockType {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        b: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::{branch::alt, bytes::complete::tag, combinator::map};

        alt((
            map(tag([0x40]), |_| BlockType::Empty),
            map(ValType::from_wasm_bytes, BlockType::Val),
            map(i64::from_wasm_bytes, |idx| {
                if !(0..=(1 << 33)).contains(&idx) {
                    todo!("we do not allow this index");
                }

                BlockType::TypeIndex(TypeIdx(idx as u32))
            }),
        ))(b)
    }
}

impl<'a> ParseWasmBinary<'a> for MemArg {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::{combinator::map, sequence::tuple};

        map(
            tuple((u32::from_wasm_bytes, u32::from_wasm_bytes)),
            |(align, offset)| MemArg(align, offset),
        )(input)
    }
}

fn control_instrs<'a, E: Debug + ParseError<Span<'a>>>(
    b: Span<'a>,
) -> nom::IResult<Span<'a>, Instr, E> {
    use nom::{
        branch::alt,
        bytes::complete::tag,
        combinator::{map, opt},
        multi::many0,
        sequence::{delimited, preceded, tuple},
    };
    alt((
        map(tag([0x00]), |_| Instr::Unreachable),
        map(tag([0x01]), |_| Instr::Nop),
        map(
            delimited(
                tag([0x02]),
                tuple((BlockType::from_wasm_bytes, many0(Instr::from_wasm_bytes))),
                tag([0x0b]),
            ),
            |(bt, ins)| Instr::Block(bt, ins.into_boxed_slice()),
        ),
        map(
            delimited(
                tag([0x03]),
                tuple((BlockType::from_wasm_bytes, many0(Instr::from_wasm_bytes))),
                tag([0x0b]),
            ),
            |(bt, ins)| Instr::Loop(bt, ins.into_boxed_slice()),
        ),
        map(
            delimited(
                tag([0x04]),
                tuple((
                    BlockType::from_wasm_bytes,
                    many0(Instr::from_wasm_bytes),
                    opt(preceded(tag([0x05]), many0(Instr::from_wasm_bytes))),
                )),
                tag([0x0b]),
            ),
            |(bt, consequent, alternate)| match alternate {
                Some(alternate) => Instr::IfElse(bt, consequent.into_boxed_slice(), alternate.into_boxed_slice()),
                None => Instr::If(bt, consequent.into_boxed_slice()),
            },
        ),
        map(preceded(tag([0x0c]), LabelIdx::from_wasm_bytes), Instr::Br),
        map(
            preceded(tag([0x0d]), LabelIdx::from_wasm_bytes),
            Instr::BrIf,
        ),
        map(
            preceded(
                tag([0x0e]),
                tuple((Vec::<LabelIdx>::from_wasm_bytes, LabelIdx::from_wasm_bytes)),
            ),
            |(tbl, alt)| Instr::BrTable(tbl.into_boxed_slice(), alt),
        ),
        map(tag([0x0f]), |_| Instr::Return),
        map(preceded(tag([0x10]), FuncIdx::from_wasm_bytes), Instr::Call),
        map(
            preceded(
                tag([0x11]),
                tuple((TypeIdx::from_wasm_bytes, TableIdx::from_wasm_bytes)),
            ),
            |(ty, tbl)| Instr::CallIndirect(ty, tbl),
        ),
    ))(b)
}

fn ref_instrs<'a, E: Debug + ParseError<Span<'a>>>(
    b: Span<'a>,
) -> nom::IResult<Span<'a>, Instr, E> {
    use nom::{branch::alt, bytes::complete::tag, combinator::map, sequence::preceded};
    alt((
        map(
            preceded(tag([0xd0]), RefType::from_wasm_bytes),
            Instr::RefNull,
        ),
        map(tag([0xd1]), |_| Instr::RefIsNull),
        map(
            preceded(tag([0xd2]), FuncIdx::from_wasm_bytes),
            Instr::RefFunc,
        ),
    ))(b)
}

fn parametric_instrs<'a, E: Debug + ParseError<Span<'a>>>(
    b: Span<'a>,
) -> nom::IResult<Span<'a>, Instr, E> {
    use nom::{branch::alt, bytes::complete::tag, combinator::map, sequence::preceded};
    alt((
        map(tag([0x1a]), |_| Instr::Drop),
        map(tag([0x1b]), |_| Instr::SelectEmpty),
        map(
            preceded(tag([0x1c]), Vec::<ValType>::from_wasm_bytes),
            |xs| Instr::Select(xs.into_boxed_slice()),
        ),
    ))(b)
}

fn variable_instrs<'a, E: Debug + ParseError<Span<'a>>>(
    b: Span<'a>,
) -> nom::IResult<Span<'a>, Instr, E> {
    use nom::{branch::alt, bytes::complete::tag, combinator::map, sequence::preceded};
    alt((
        map(
            preceded(tag([0x20]), LocalIdx::from_wasm_bytes),
            Instr::LocalGet,
        ),
        map(
            preceded(tag([0x21]), LocalIdx::from_wasm_bytes),
            Instr::LocalSet,
        ),
        map(
            preceded(tag([0x22]), LocalIdx::from_wasm_bytes),
            Instr::LocalTee,
        ),
        map(
            preceded(tag([0x23]), GlobalIdx::from_wasm_bytes),
            Instr::GlobalGet,
        ),
        map(
            preceded(tag([0x24]), GlobalIdx::from_wasm_bytes),
            Instr::GlobalSet,
        ),
    ))(b)
}

fn table_instrs<'a, E: Debug + ParseError<Span<'a>>>(
    b: Span<'a>,
) -> nom::IResult<Span<'a>, Instr, E> {
    use nom::{
        branch::alt,
        bytes::complete::tag,
        combinator::map,
        sequence::preceded,
    };

    alt((
        map(
            preceded(tag([0x25]), TableIdx::from_wasm_bytes),
            Instr::TableGet,
        ),
        map(
            preceded(tag([0x26]), TableIdx::from_wasm_bytes),
            Instr::TableSet,
        ),
    ))(b)
}

fn memory_instrs<'a, E: Debug + ParseError<Span<'a>>>(
    input: Span<'a>,
) -> nom::IResult<Span<'a>, Instr, E> {
    use nom::{
        branch::alt,
        bytes::complete::tag,
        combinator::map,
        sequence::{preceded, tuple},
    };

    alt((
        alt((
            map(
                preceded(tag([0x28]), MemArg::from_wasm_bytes),
                Instr::I32Load,
            ),
            map(
                preceded(tag([0x29]), MemArg::from_wasm_bytes),
                Instr::I64Load,
            ),
            map(
                preceded(tag([0x2a]), MemArg::from_wasm_bytes),
                Instr::F32Load,
            ),
            map(
                preceded(tag([0x2b]), MemArg::from_wasm_bytes),
                Instr::F64Load,
            ),
            map(
                preceded(tag([0x2c]), MemArg::from_wasm_bytes),
                Instr::I32Load8S,
            ),
            map(
                preceded(tag([0x2d]), MemArg::from_wasm_bytes),
                Instr::I32Load8U,
            ),
            map(
                preceded(tag([0x2e]), MemArg::from_wasm_bytes),
                Instr::I32Load16S,
            ),
            map(
                preceded(tag([0x2f]), MemArg::from_wasm_bytes),
                Instr::I32Load16U,
            ),
            map(
                preceded(tag([0x30]), MemArg::from_wasm_bytes),
                Instr::I64Load8S,
            ),
            map(
                preceded(tag([0x31]), MemArg::from_wasm_bytes),
                Instr::I64Load8U,
            ),
            map(
                preceded(tag([0x32]), MemArg::from_wasm_bytes),
                Instr::I64Load16S,
            ),
        )),
        alt((
            map(
                preceded(tag([0x33]), MemArg::from_wasm_bytes),
                Instr::I64Load16U,
            ),
            map(
                preceded(tag([0x34]), MemArg::from_wasm_bytes),
                Instr::I64Load32S,
            ),
            map(
                preceded(tag([0x35]), MemArg::from_wasm_bytes),
                Instr::I64Load32U,
            ),
            map(
                preceded(tag([0x36]), MemArg::from_wasm_bytes),
                Instr::I32Store,
            ),
            map(
                preceded(tag([0x37]), MemArg::from_wasm_bytes),
                Instr::I64Store,
            ),
            map(
                preceded(tag([0x38]), MemArg::from_wasm_bytes),
                Instr::F32Store,
            ),
            map(
                preceded(tag([0x39]), MemArg::from_wasm_bytes),
                Instr::F64Store,
            ),
            map(
                preceded(tag([0x3a]), MemArg::from_wasm_bytes),
                Instr::I32Store8,
            ),
            map(
                preceded(tag([0x3b]), MemArg::from_wasm_bytes),
                Instr::I32Store16,
            ),
            map(
                preceded(tag([0x3c]), MemArg::from_wasm_bytes),
                Instr::I64Store8,
            ),
            map(
                preceded(tag([0x3d]), MemArg::from_wasm_bytes),
                Instr::I64Store16,
            ),
            map(
                preceded(tag([0x3e]), MemArg::from_wasm_bytes),
                Instr::I64Store32,
            ),
        )),
        map(
            preceded(tag([0x3f]), MemIdx::from_wasm_bytes),
            Instr::MemorySize,
        ),
        map(
            preceded(tag([0x40]), MemIdx::from_wasm_bytes),
            Instr::MemoryGrow,
        ),
        // TODO: these should be encoded as u32s, but instead I'm just... pulling
        // them in as byte literals, since they're all <2 bytes long in LEB128.
    ))(input)
}

fn numeric_instrs<'a, E: Debug + ParseError<Span<'a>>>(
    input: Span<'a>,
) -> nom::IResult<Span<'a>, Instr, E> {
    use nom::{branch::alt, bytes::complete::tag, combinator::map, sequence::preceded};

    alt((
        alt((
            map(preceded(tag([0x41]), i32::from_wasm_bytes), Instr::I32Const),
            map(preceded(tag([0x42]), i64::from_wasm_bytes), Instr::I64Const),
            map(preceded(tag([0x43]), f32::from_wasm_bytes), Instr::F32Const),
            map(preceded(tag([0x44]), f64::from_wasm_bytes), Instr::F64Const),
        )),
        alt((
            map(tag([0x45]), |_| Instr::I32Eqz),
            map(tag([0x46]), |_| Instr::I32Eq),
            map(tag([0x47]), |_| Instr::I32Ne),
            map(tag([0x48]), |_| Instr::I32LtS),
            map(tag([0x49]), |_| Instr::I32LtU),
            map(tag([0x4a]), |_| Instr::I32GtS),
            map(tag([0x4b]), |_| Instr::I32GtU),
            map(tag([0x4c]), |_| Instr::I32LeS),
            map(tag([0x4d]), |_| Instr::I32LeU),
            map(tag([0x4e]), |_| Instr::I32GeS),
            map(tag([0x4f]), |_| Instr::I32GeU),
            map(tag([0x50]), |_| Instr::I64Eqz),
            map(tag([0x51]), |_| Instr::I64Eq),
            map(tag([0x52]), |_| Instr::I64Ne),
            map(tag([0x53]), |_| Instr::I64LtS),
            map(tag([0x54]), |_| Instr::I64LtU),
            map(tag([0x55]), |_| Instr::I64GtS),
        )),
        alt((
            map(tag([0x56]), |_| Instr::I64GtU),
            map(tag([0x57]), |_| Instr::I64LeS),
            map(tag([0x58]), |_| Instr::I64LeU),
            map(tag([0x59]), |_| Instr::I64GeS),
            map(tag([0x5a]), |_| Instr::I64GeU),
            map(tag([0x5b]), |_| Instr::F32Eq),
            map(tag([0x5c]), |_| Instr::F32Ne),
            map(tag([0x5d]), |_| Instr::F32Lt),
            map(tag([0x5e]), |_| Instr::F32Gt),
            map(tag([0x5f]), |_| Instr::F32Le),
            map(tag([0x60]), |_| Instr::F32Ge),
            map(tag([0x61]), |_| Instr::F64Eq),
            map(tag([0x62]), |_| Instr::F64Ne),
            map(tag([0x63]), |_| Instr::F64Lt),
            map(tag([0x64]), |_| Instr::F64Gt),
            map(tag([0x65]), |_| Instr::F64Le),
            map(tag([0x66]), |_| Instr::F64Ge),
        )),
        alt((
            map(tag([0x67]), |_| Instr::I32Clz),
            map(tag([0x68]), |_| Instr::I32Ctz),
            map(tag([0x69]), |_| Instr::I32Popcnt),
            map(tag([0x6a]), |_| Instr::I32Add),
            map(tag([0x6b]), |_| Instr::I32Sub),
            map(tag([0x6c]), |_| Instr::I32Mul),
            map(tag([0x6d]), |_| Instr::I32DivS),
            map(tag([0x6e]), |_| Instr::I32DivU),
            map(tag([0x6f]), |_| Instr::I32RemS),
            map(tag([0x70]), |_| Instr::I32RemU),
            map(tag([0x71]), |_| Instr::I32And),
            map(tag([0x72]), |_| Instr::I32Ior),
            map(tag([0x73]), |_| Instr::I32Xor),
            map(tag([0x74]), |_| Instr::I32Shl),
            map(tag([0x75]), |_| Instr::I32ShrS),
            map(tag([0x76]), |_| Instr::I32ShrU),
            map(tag([0x77]), |_| Instr::I32Rol),
            map(tag([0x78]), |_| Instr::I32Ror),
            map(tag([0x79]), |_| Instr::I64Clz),
            map(tag([0x7a]), |_| Instr::I64Ctz),
        )),
        alt((
            map(tag([0x7b]), |_| Instr::I64Popcnt),
            map(tag([0x7c]), |_| Instr::I64Add),
            map(tag([0x7d]), |_| Instr::I64Sub),
            map(tag([0x7e]), |_| Instr::I64Mul),
            map(tag([0x7f]), |_| Instr::I64DivS),
            map(tag([0x80]), |_| Instr::I64DivU),
            map(tag([0x81]), |_| Instr::I64RemS),
            map(tag([0x82]), |_| Instr::I64RemU),
            map(tag([0x83]), |_| Instr::I64And),
            map(tag([0x84]), |_| Instr::I64Ior),
            map(tag([0x85]), |_| Instr::I64Xor),
            map(tag([0x86]), |_| Instr::I64Shl),
            map(tag([0x87]), |_| Instr::I64ShrS),
            map(tag([0x88]), |_| Instr::I64ShrU),
            map(tag([0x89]), |_| Instr::I64Rol),
            map(tag([0x8a]), |_| Instr::I64Ror),
            map(tag([0x8b]), |_| Instr::F32Abs),
            map(tag([0x8c]), |_| Instr::F32Neg),
            map(tag([0x8d]), |_| Instr::F32Ceil),
            map(tag([0x8e]), |_| Instr::F32Floor),
        )),
        alt((
            map(tag([0x8f]), |_| Instr::F32Trunc),
            map(tag([0x90]), |_| Instr::F32NearestInt),
            map(tag([0x91]), |_| Instr::F32Sqrt),
            map(tag([0x92]), |_| Instr::F32Add),
            map(tag([0x93]), |_| Instr::F32Sub),
            map(tag([0x94]), |_| Instr::F32Mul),
            map(tag([0x95]), |_| Instr::F32Div),
            map(tag([0x96]), |_| Instr::F32Min),
            map(tag([0x97]), |_| Instr::F32Max),
            map(tag([0x98]), |_| Instr::F32CopySign),
            map(tag([0x99]), |_| Instr::F64Abs),
            map(tag([0x9a]), |_| Instr::F64Neg),
            map(tag([0x9b]), |_| Instr::F64Ceil),
            map(tag([0x9c]), |_| Instr::F64Floor),
            map(tag([0x9d]), |_| Instr::F64Trunc),
            map(tag([0x9e]), |_| Instr::F64NearestInt),
            map(tag([0x9f]), |_| Instr::F64Sqrt),
        )),
        alt((
            map(tag([0xa0]), |_| Instr::F64Add),
            map(tag([0xa1]), |_| Instr::F64Sub),
            map(tag([0xa2]), |_| Instr::F64Mul),
            map(tag([0xa3]), |_| Instr::F64Div),
            map(tag([0xa4]), |_| Instr::F64Min),
            map(tag([0xa5]), |_| Instr::F64Max),
            map(tag([0xa6]), |_| Instr::F64CopySign),
            map(tag([0xa7]), |_| Instr::I32ConvertI64),
            map(tag([0xa8]), |_| Instr::I32SConvertF32),
            map(tag([0xa9]), |_| Instr::I32UConvertF32),
            map(tag([0xaa]), |_| Instr::I32SConvertF64),
            map(tag([0xab]), |_| Instr::I32UConvertF64),
            map(tag([0xac]), |_| Instr::I64SConvertI32),
            map(tag([0xad]), |_| Instr::I64UConvertI32),
            map(tag([0xae]), |_| Instr::I64SConvertF32),
            map(tag([0xaf]), |_| Instr::I64UConvertF32),
            map(tag([0xb0]), |_| Instr::I64SConvertF64),
        )),
        alt((
            map(tag([0xb1]), |_| Instr::I64UConvertF64),
            map(tag([0xb2]), |_| Instr::F32SConvertI32),
            map(tag([0xb3]), |_| Instr::F32UConvertI32),
            map(tag([0xb4]), |_| Instr::F32SConvertI64),
            map(tag([0xb5]), |_| Instr::F32UConvertI64),
            map(tag([0xb6]), |_| Instr::F32ConvertF64),
            map(tag([0xb7]), |_| Instr::F64SConvertI32),
            map(tag([0xb8]), |_| Instr::F64UConvertI32),
            map(tag([0xb9]), |_| Instr::F64SConvertI64),
            map(tag([0xba]), |_| Instr::F64UConvertI64),
            map(tag([0xbb]), |_| Instr::F64ConvertF32),
            map(tag([0xbc]), |_| Instr::I32ReinterpretF32),
            map(tag([0xbd]), |_| Instr::I64ReinterpretF64),
            map(tag([0xbe]), |_| Instr::F32ReinterpretI32),
            map(tag([0xbf]), |_| Instr::F64ReinterpretI64),
            map(tag([0xc0]), |_| Instr::I32SExtendI8),
        )),
        alt((
            map(tag([0xc1]), |_| Instr::I32SExtendI16),
            map(tag([0xc2]), |_| Instr::I64SExtendI8),
            map(tag([0xc3]), |_| Instr::I64SExtendI16),
            map(tag([0xc4]), |_| Instr::I64SExtendI32),
        )),
    ))(input)
}

fn multibyte_instrs<'a, E: Debug + ParseError<Span<'a>>>(
    input: Span<'a>,
) -> nom::IResult<Span<'a>, Instr, E> {
    use nom::{bytes::complete::tag, branch::alt, sequence::tuple, combinator::{ fail, map }};
    let (input, t) = alt((
        tag([0xfc]),
        tag([0xfd]),
        tag([0xfe]),
        tag([0xff])
    ))(input)?;

    let (input, mb) = u32::from_wasm_bytes(input)?;

    let instr = match (t[0], mb) {
        (0xfc, 0x00) => Instr::I32SConvertSatF32,
        (0xfc, 0x01) => Instr::I32UConvertSatF32,
        (0xfc, 0x02) => Instr::I32SConvertSatF64,
        (0xfc, 0x03) => Instr::I32UConvertSatF64,
        (0xfc, 0x04) => Instr::I64SConvertSatF32,
        (0xfc, 0x05) => Instr::I64UConvertSatF32,
        (0xfc, 0x06) => Instr::I64SConvertSatF64,
        (0xfc, 0x07) => Instr::I64UConvertSatF64,
        (0xfc, 0x08) => {
            return map(
                tuple((DataIdx::from_wasm_bytes, MemIdx::from_wasm_bytes)),
                |(di, mi)| Instr::MemoryInit(di, mi)
            )(input)
        },
        (0xfc, 0x09) => {
            return map(
                DataIdx::from_wasm_bytes,
                Instr::DataDrop,
            )(input)
        },
        (0xfc, 0x0a) => {
            return map(
                tuple((MemIdx::from_wasm_bytes, MemIdx::from_wasm_bytes)),
                |(mi0, mi1)| Instr::MemoryCopy(mi0, mi1),
            )(input)
        }
        (0xfc, 0x0b) => {
            return map(
                MemIdx::from_wasm_bytes,
                Instr::MemoryFill,
            )(input)
        }
        (0xfc, 0x0c) => {
            return map(
                tuple((ElemIdx::from_wasm_bytes, TableIdx::from_wasm_bytes)),
                |(ei, ti)| Instr::TableInit(ei, ti)
            )(input)
        }
        (0xfc, 0x0d) => {
            return map(
                ElemIdx::from_wasm_bytes,
                Instr::ElemDrop,
            )(input)
        }
        (0xfc, 0x0e) => {
            return map(
                tuple((TableIdx::from_wasm_bytes, TableIdx::from_wasm_bytes)),
                |(ei, ti)| Instr::TableCopy(ei, ti),
            )(input)
        }
        (0xfc, 0x0f) => {
            return map(
                TableIdx::from_wasm_bytes,
                Instr::TableGrow
            )(input)
        },
        (0xfc, 0x10) => {
            return map(
                TableIdx::from_wasm_bytes,
                Instr::TableSize
            )(input)
        }
        (0xfc, 0x11) => {
            return map(
                TableIdx::from_wasm_bytes,
                Instr::TableFill
            )(input)
        }

        _ => return fail::<_, Instr, _>(input)
    };

    Ok((input, instr))
}


impl<'a> ParseWasmBinary<'a> for Instr {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        b: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::branch::alt;

        alt((
            control_instrs,
            ref_instrs,
            parametric_instrs,
            variable_instrs,
            table_instrs,
            memory_instrs,
            numeric_instrs,
            multibyte_instrs,
        ))(b)
    }
}

impl<'a> ParseWasmBinary<'a> for Expr {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        b: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::{bytes::complete::tag, combinator::map, multi::many0, sequence::terminated};

        map(terminated(many0(Instr::from_wasm_bytes), tag([0x0b])), Expr)(b)
    }
}

impl_parse_for_newtype!(TypeIdx, u32);
impl_parse_for_newtype!(FuncIdx, u32);
impl_parse_for_newtype!(TableIdx, u32);
impl_parse_for_newtype!(MemIdx, u32);
impl_parse_for_newtype!(GlobalIdx, u32);
impl_parse_for_newtype!(ElemIdx, u32);
impl_parse_for_newtype!(DataIdx, u32);
impl_parse_for_newtype!(LocalIdx, u32);
impl_parse_for_newtype!(LabelIdx, u32);

impl<'a> ParseWasmBinary<'a> for Import<'a> {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::{combinator::map, sequence::tuple};

        map(
            tuple((
                Name::from_wasm_bytes,
                Name::from_wasm_bytes,
                ImportDesc::from_wasm_bytes,
            )),
            |(r#mod, nm, desc)| Import { r#mod, nm, desc },
        )(input)
    }
}

impl<'a> ParseWasmBinary<'a> for ImportDesc {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::{branch::alt, bytes::complete::tag, combinator::map, sequence::preceded};

        alt((
            map(
                preceded(tag([0x00]), TypeIdx::from_wasm_bytes),
                ImportDesc::Func,
            ),
            map(
                preceded(tag([0x01]), TableType::from_wasm_bytes),
                ImportDesc::Table,
            ),
            map(
                preceded(tag([0x02]), MemType::from_wasm_bytes),
                ImportDesc::Mem,
            ),
            map(
                preceded(tag([0x03]), GlobalType::from_wasm_bytes),
                ImportDesc::Global,
            ),
        ))(input)
    }
}

impl<'a> ParseWasmBinary<'a> for Export<'a> {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::{combinator::map, sequence::tuple};

        map(
            tuple((Name::from_wasm_bytes, ExportDesc::from_wasm_bytes)),
            |(nm, desc)| Export { nm, desc },
        )(input)
    }
}

impl<'a> ParseWasmBinary<'a> for ExportDesc {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::{branch::alt, bytes::complete::tag, combinator::map, sequence::preceded};

        alt((
            map(
                preceded(tag([0x00]), FuncIdx::from_wasm_bytes),
                ExportDesc::Func,
            ),
            map(
                preceded(tag([0x01]), TableIdx::from_wasm_bytes),
                ExportDesc::Table,
            ),
            map(
                preceded(tag([0x02]), MemIdx::from_wasm_bytes),
                ExportDesc::Mem,
            ),
            map(
                preceded(tag([0x03]), GlobalIdx::from_wasm_bytes),
                ExportDesc::Global,
            ),
        ))(input)
    }
}

impl<'a> ParseWasmBinary<'a> for Elem {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::{combinator::map, sequence::tuple, multi::many0};

        let (input, pos) = nom_locate::position(input)?;
        let (input, flags) = u32::from_wasm_bytes(input)?;

        let pos = pos.location_offset();
        /*
        ┌─── element type+exprs vs element kind + element idx
        │┌── explicit table index (or distinguishes passive from declarative)
        ││┌─ Passive or Declarative
        ↓↓↓
        000: expr vec<funcidx>                      -> active
        001: elemkind vec<funcidx>                  -> passive
        010: tableidx expr elemkind vec<funcidx>    -> active
        011: elemkind vec<funcidx>                  -> declarative
        100: expr vec<expr>                         -> active
        101: reftype vec<expr>                      -> passive
        110: tableidx expr reftype vec<expr>        -> active
        111: reftype vec<expr>                      -> declarative
        */

        match flags & 7 {
            0b000 => map(
                tuple((Expr::from_wasm_bytes, Vec::<FuncIdx>::from_wasm_bytes)),
                |(ex, fs)| Elem::ActiveSegmentFuncs(ex, fs),
            )(input),
            0b001 => map(
                tuple((u32::from_wasm_bytes, Vec::<FuncIdx>::from_wasm_bytes)),
                |(el, fs)| Elem::PassiveSegment(el, fs),
            )(input),
            0b010 => map(
                tuple((
                    TableIdx::from_wasm_bytes,
                    Expr::from_wasm_bytes,
                    u32::from_wasm_bytes,
                    Vec::<FuncIdx>::from_wasm_bytes,
                )),
                |(a, b, c, d)| Elem::ActiveSegment(a, b, c, d),
            )(input),
            0b011 => map(
                tuple((u32::from_wasm_bytes, Vec::<FuncIdx>::from_wasm_bytes)),
                |(el, fs)| Elem::DeclarativeSegment(el, fs),
            )(input),
            0b100 => map(
                tuple((Expr::from_wasm_bytes, Vec::<Expr>::from_wasm_bytes)),
                |(ex, es)| Elem::ActiveSegmentExpr(ex, es),
            )(input),
            0b101 => map(
                tuple((RefType::from_wasm_bytes, Vec::<Expr>::from_wasm_bytes)),
                |(rt, es)| Elem::PassiveSegmentExpr(rt, es),
            )(input),
            0b110 => map(
                tuple((
                    TableIdx::from_wasm_bytes,
                    Expr::from_wasm_bytes,
                    RefType::from_wasm_bytes,
                    many0(Expr::from_wasm_bytes),
                )),
                |(a, b, c, d)| Elem::ActiveSegmentTableAndExpr(a, b, c, d),
            )(input),
            0b111 => map(
                tuple((RefType::from_wasm_bytes, Vec::<Expr>::from_wasm_bytes)),
                |(rt, es)| Elem::DeclarativeSegmentExpr(rt, es),
            )(input),
            _ => unreachable!(),
        }
    }
}

impl<'a> ParseWasmBinary<'a> for Local {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::{combinator::map, sequence::tuple};

        map(
            tuple((u32::from_wasm_bytes, ValType::from_wasm_bytes)),
            |(x, y)| Local(x, y),
        )(input)
    }
}

impl<'a> ParseWasmBinary<'a> for Func {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::{combinator::map, sequence::tuple};

        map(
            tuple((Vec::<Local>::from_wasm_bytes, Expr::from_wasm_bytes)),
            |(locals, expr)| Func { locals, expr },
        )(input)
    }
}

impl<'a> ParseWasmBinary<'a> for Code {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::{bytes::complete::take, combinator::fail};

        let (input, size) = u32::from_wasm_bytes(input)?;
        let (input, section) = take(size as usize)(input)?;

        let (remaining, func) = Func::from_wasm_bytes(section)?;
        if !remaining.is_empty() {
            return fail::<_, Self, _>(input);
        }

        Ok((input, Code(func)))
    }
}

impl<'a> ParseWasmBinary<'a> for Data<'a> {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::{
            combinator::{fail, map},
            sequence::tuple,
        };

        let (input, value) = u32::from_wasm_bytes(input)?;

        match value {
            0x00 => map(tuple((Expr::from_wasm_bytes, ByteVec::from_wasm_bytes)), |(expr, bvec)| Data::Active(bvec, MemIdx(0), expr))(input),
            0x01 => map(ByteVec::from_wasm_bytes, Data::Passive)(input),
            0x02 => map(tuple((
                        MemIdx::from_wasm_bytes,
                        Expr::from_wasm_bytes,
                        ByteVec::from_wasm_bytes,
                    )), |(memidx, expr, bvec)| Data::Active(bvec, memidx, expr))(input),
            _ => fail::<_, Self, _>(input)
        }
    }
}

impl<'a> ParseWasmBinary<'a> for SectionType<'a> {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::{bytes::complete::take, combinator::fail, sequence::pair};

        let (input, (section_id, size)) = pair(take(1usize), u32::from_wasm_bytes)(input)?;
        let (input, section) = take(size as usize)(input)?;

        let section = match section_id[0] {
            0x0 => SectionType::Custom(&section[..]),
            0x1 => SectionType::Type(Vec::<Type>::from_wasm_bytes(section)?.1),
            0x2 => SectionType::Import(Vec::<Import>::from_wasm_bytes(section)?.1),
            0x3 => SectionType::Function(Vec::<TypeIdx>::from_wasm_bytes(section)?.1),
            0x4 => SectionType::Table(Vec::<TableType>::from_wasm_bytes(section)?.1),
            0x5 => SectionType::Memory(Vec::<MemType>::from_wasm_bytes(section)?.1),
            0x6 => SectionType::Global(Vec::<Global>::from_wasm_bytes(section)?.1),
            0x7 => SectionType::Export(Vec::<Export>::from_wasm_bytes(section)?.1),
            0x8 => SectionType::Start(FuncIdx::from_wasm_bytes(section)?.1),
            0x9 => SectionType::Element(Vec::<Elem>::from_wasm_bytes(section)?.1),
            0xa => SectionType::Code(Vec::<Code>::from_wasm_bytes(section)?.1),
            0xb => SectionType::Data(Vec::<Data>::from_wasm_bytes(section)?.1),
            0xc => SectionType::DataCount(u32::from_wasm_bytes(section)?.1),
            _ => return fail::<_, Self, _>(section),
        };

        Ok((input, section))
    }
}

impl<'a> ParseWasmBinary<'a> for Module<'a> {
    fn from_wasm_bytes<E: Debug + ParseError<Span<'a>>>(
        input: Span<'a>,
    ) -> nom::IResult<Span<'a>, Self, E> {
        use nom::{bytes::complete::tag, combinator::fail};

        let (mut input, _) = tag([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x0, 0x0])(input)?;

        let mut module_builder = ModuleBuilder::new();
        while !input.is_empty() {
            let section;
            (input, section) = SectionType::from_wasm_bytes(input)?;

            match section {
                SectionType::Custom(xs) => module_builder = module_builder.custom_section(xs),

                SectionType::Type(mut xs) => {
                    xs.shrink_to_fit();
                    module_builder = module_builder.type_section(xs);
                }

                SectionType::Import(mut xs) => {
                    xs.shrink_to_fit();
                    module_builder = module_builder.import_section(xs);
                }

                SectionType::Function(mut xs) => {
                    xs.shrink_to_fit();
                    module_builder = module_builder.function_section(xs);
                }

                SectionType::Table(mut xs) => {
                    xs.shrink_to_fit();
                    module_builder = module_builder.table_section(xs);
                }

                SectionType::Memory(mut xs) => {
                    xs.shrink_to_fit();
                    module_builder = module_builder.memory_section(xs);
                }

                SectionType::Global(mut xs) => {
                    xs.shrink_to_fit();
                    module_builder = module_builder.global_section(xs);
                }

                SectionType::Export(mut xs) => {
                    xs.shrink_to_fit();
                    module_builder = module_builder.export_section(xs);
                }

                SectionType::Start(xs) => {
                    module_builder = module_builder.start_section(xs);
                }

                SectionType::Element(mut xs) => {
                    xs.shrink_to_fit();
                    module_builder = module_builder.element_section(xs);
                }

                SectionType::Code(mut xs) => {
                    xs.shrink_to_fit();
                    module_builder = module_builder.code_section(xs);
                }

                SectionType::Data(mut xs) => {
                    xs.shrink_to_fit();
                    module_builder = module_builder.data_section(xs);
                }

                SectionType::DataCount(xs) => {
                    module_builder = module_builder.datacount_section(xs);
                }

                _ => return fail::<_, Self, _>(input),
            }
        }

        let module = module_builder.build();
        Ok((input, module))
    }
}

pub(crate) fn parse(input: &[u8]) -> anyhow::Result<Module> {
    match Module::from_wasm_bytes::<nom::error::VerboseError<Span>>(Span::new(input)) {
        Ok((_, wasm)) => Ok(wasm),
        Err(err) => {
            anyhow::bail!("{:?}", err)
        }
    }
}

#[cfg(test)]
mod test {
    use std::io::Write;

    use super::*;

    #[test]
    fn test_read_u32() {
        use leb128;
        let mut v = Vec::new();
        leb128::write::unsigned(&mut v, 8082008).unwrap();
        let (rest, v) = u32::from_wasm_bytes::<nom::error::Error<Span>>(Span::new(&v[..])).unwrap();
        assert_eq!(v, 8082008);
        assert_eq!(&rest[..], b"");

        let mut v = Vec::new();
        leb128::write::unsigned(&mut v, u32::MAX as u64).unwrap();
        let (rest, v) = u32::from_wasm_bytes::<nom::error::Error<Span>>(Span::new(&v[..])).unwrap();
        assert_eq!(v, u32::MAX);
        assert_eq!(&rest[..], b"");

        let mut v = Vec::new();
        leb128::write::unsigned(&mut v, u32::MIN as u64).unwrap();
        let (rest, v) = u32::from_wasm_bytes::<nom::error::Error<Span>>(Span::new(&v[..])).unwrap();
        assert_eq!(v, u32::MIN);
        assert_eq!(&rest[..], b"");

        let mut v = Vec::new();
        leb128::write::unsigned(&mut v, 0).unwrap();
        let (rest, v) = u32::from_wasm_bytes::<nom::error::Error<Span>>(Span::new(&v[..])).unwrap();
        assert_eq!(v, 0);
        assert_eq!(&rest[..], b"");
    }

    #[test]
    fn test_read_i32() {
        use leb128;
        let mut v = Vec::new();
        leb128::write::signed(&mut v, 8082008).unwrap();
        let (rest, v) = i32::from_wasm_bytes::<nom::error::Error<Span>>(Span::new(&v[..])).unwrap();
        assert_eq!(v, 8082008);
        assert_eq!(&rest[..], b"");

        let mut v = Vec::new();
        leb128::write::signed(&mut v, -45).unwrap();
        let (rest, v) = i32::from_wasm_bytes::<nom::error::Error<Span>>(Span::new(&v[..])).unwrap();
        assert_eq!(v, -45);
        assert_eq!(&rest[..], b"");

        let mut v = Vec::new();
        leb128::write::signed(&mut v, i32::MAX as i64).unwrap();
        let (rest, v) = i32::from_wasm_bytes::<nom::error::Error<Span>>(Span::new(&v[..])).unwrap();
        assert_eq!(v, i32::MAX);
        assert_eq!(&rest[..], b"");

        let mut v = Vec::new();
        leb128::write::signed(&mut v, i32::MIN as i64).unwrap();
        let (rest, v) = i32::from_wasm_bytes::<nom::error::Error<Span>>(Span::new(&v[..])).unwrap();
        assert_eq!(v, i32::MIN);
        assert_eq!(&rest[..], b"");

        let mut v = Vec::new();
        leb128::write::signed(&mut v, 0).unwrap();
        let (rest, v) = i32::from_wasm_bytes::<nom::error::Error<Span>>(Span::new(&v[..])).unwrap();
        assert_eq!(v, 0);
        assert_eq!(&rest[..], b"");
    }

    #[test]
    fn test_read_vec_i64() {
        use leb128;
        let mut v = Vec::new();
        leb128::write::unsigned(&mut v, 3).unwrap();
        leb128::write::signed(&mut v, 1).unwrap();
        leb128::write::signed(&mut v, 1).unwrap();
        leb128::write::signed(&mut v, 2).unwrap();
        leb128::write::signed(&mut v, 3).unwrap();
        leb128::write::signed(&mut v, 5).unwrap();
        let (rest, v) =
            Vec::<i64>::from_wasm_bytes::<nom::error::Error<Span>>(Span::new(&v[..])).unwrap();
        assert_eq!(v, vec![1, 1, 2]);
        assert_eq!(&rest[..], [3, 5]);
    }

    #[test]
    fn test_read_str() {
        use leb128;
        let xs = "hello world!";
        let mut v = Vec::new();
        leb128::write::unsigned(&mut v, xs.len() as u64).unwrap();
        write!(&mut v, "{}", xs).unwrap();

        let (rest, v) =
            <&str>::from_wasm_bytes::<nom::error::Error<Span>>(Span::new(&v[..])).unwrap();
        assert_eq!(v, "hello world!");
        assert_eq!(&rest[..], &[] as &[u8]);
    }

    #[test]
    fn test_read_wasm() {
        let bytes = include_bytes!("../example.wasm");

        let (input, wasm) =
            Module::from_wasm_bytes::<nom::error::VerboseError<Span>>(Span::new(bytes)).unwrap();

        assert!(input.is_empty());

        let cmp = ModuleBuilder::new()
            .type_section(vec![Type(
                ResultType(vec![
                    ValType::NumType(NumType::I32),
                    ValType::NumType(NumType::I32),
                ]),
                ResultType(vec![ValType::NumType(NumType::I32)]),
            )])
            .function_section(vec![TypeIdx(0)])
            .memory_section(vec![MemType(Limits::Min(64))])
            .export_section(vec![Export {
                nm: Name("add_i32"),
                desc: ExportDesc::Func(FuncIdx(0)),
            }])
            .code_section(vec![Code(Func {
                locals: vec![],
                expr: Expr(vec![
                    Instr::LocalGet(LocalIdx(0)),
                    Instr::LocalGet(LocalIdx(1)),
                    Instr::I32Add,
                ]),
            })])
            .build();

        // XXX future chris, turn this into a struct with fields instead of a struct with one big
        // "sections" vec
        assert_eq!(wasm, cmp);
    }
}
