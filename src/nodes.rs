#![allow(dead_code)]

use divrem::DivCeil;
use nom_locate::LocatedSpan;

#[derive(Debug)]
struct ByteVec(Vec<u8>);
#[derive(Debug)]
struct Name(String);

type Span<'a> = LocatedSpan<&'a [u8]>;

// A TODO struct that borrows from journalism [1]
// [1]: https://en.wikipedia.org/wiki/To_come_(publishing)
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
struct TKTK;

macro_rules! impl_parse_for_newtype {
    ($type:ident, $innertype:ident) => {
        impl ParseWasmBinary for $type {
            fn from_wasm_bytes(input: Span<'_>) -> nom::IResult<Span<'_>, Self> {
                use nom::combinator::map;
                map($innertype::from_wasm_bytes, $type)(input)
            }
        }
    };
}

trait ParseWasmBinary: Sized {
    fn from_wasm_bytes(b: Span<'_>) -> nom::IResult<Span<'_>, Self>;
}

impl<T: ParseWasmBinary> ParseWasmBinary for Vec<T> {
    fn from_wasm_bytes(b: Span<'_>) -> nom::IResult<Span<'_>, Self> {
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

impl ParseWasmBinary for ByteVec {
    fn from_wasm_bytes(input: Span<'_>) -> nom::IResult<Span<'_>, Self> {
        use nom::bytes::complete::take;
        let (input, sz) = <u32 as ParseWasmBinary>::from_wasm_bytes(input)?;
        let (input, span) = take(sz as usize)(input)?;
        Ok((input, ByteVec(span.to_vec())))
    }
}

impl ParseWasmBinary for String {
    fn from_wasm_bytes(input: Span<'_>) -> nom::IResult<Span<'_>, Self> {
        use nom::bytes::complete::take;
        let (input, sz) = <u32 as ParseWasmBinary>::from_wasm_bytes(input)?;
        let (input, span) = take(sz as usize)(input)?;
        let Ok(xs) = std::str::from_utf8(&span) else {
            return nom::combinator::fail(input);
        };

        Ok((input, xs.to_string()))
    }
}

impl ParseWasmBinary for Name {
    fn from_wasm_bytes(input: Span<'_>) -> nom::IResult<Span<'_>, Self> {
        let (input, sz) = String::from_wasm_bytes(input)?;

        Ok((input, Name(sz)))
    }
}

macro_rules! parse_leb128 {
    ($type:ident, signed) => {
        impl ParseWasmBinary for $type {
            fn from_wasm_bytes(input: Span<'_>) -> nom::IResult<Span<'_>, Self> {
                use nom::bytes::complete::{take, take_till};

                let (input, leb_bytes) = take_till(|xs| xs & 0x80 == 0)(input)?;
                let (input, last) = take(1usize)(input)?;

                if leb_bytes.len() + 1 > $type::BITS.div_ceil(7) as usize {
                    todo!("return fatal error: overflow");
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
        impl ParseWasmBinary for $type {
            fn from_wasm_bytes(input: Span<'_>) -> nom::IResult<Span<'_>, Self> {
                use nom::bytes::complete::{take, take_till};

                let (input, leb_bytes) = take_till(|xs| xs & 0x80 == 0)(input)?;
                let (input, last) = take(1usize)(input)?;

                if leb_bytes.len() + 1 > $type::BITS.div_ceil(7) as usize {
                    todo!("return fatal error: overflow");
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

impl ParseWasmBinary for f32 {
    fn from_wasm_bytes(b: Span<'_>) -> nom::IResult<Span<'_>, Self> {
        use nom::number::complete::le_f32;
        le_f32(b)
    }
}

impl ParseWasmBinary for f64 {
    fn from_wasm_bytes(b: Span<'_>) -> nom::IResult<Span<'_>, Self> {
        use nom::number::complete::le_f64;
        le_f64(b)
    }
}

#[derive(Debug)]
enum NumType {
    I32,
    I64,
    F32,
    F64,
}

impl ParseWasmBinary for NumType {
    fn from_wasm_bytes(input: Span<'_>) -> nom::IResult<Span<'_>, Self> {
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

#[derive(Debug)]
enum VecType {
    V128,
}

impl ParseWasmBinary for VecType {
    fn from_wasm_bytes(input: Span<'_>) -> nom::IResult<Span<'_>, Self> {
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

#[derive(Debug)]
enum RefType {
    FuncRef,
    ExternRef,
}

impl ParseWasmBinary for RefType {
    fn from_wasm_bytes(input: Span<'_>) -> nom::IResult<Span<'_>, Self> {
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

#[allow(clippy::enum_variant_names)]
#[derive(Debug)]
enum ValType {
    NumType(NumType),
    VecType(VecType),
    RefType(RefType),
}

impl ParseWasmBinary for ValType {
    fn from_wasm_bytes(input: Span<'_>) -> nom::IResult<Span<'_>, Self> {
        use nom::{branch::alt, combinator::map};

        alt((
            map(NumType::from_wasm_bytes, ValType::NumType),
            map(VecType::from_wasm_bytes, ValType::VecType),
            map(RefType::from_wasm_bytes, ValType::RefType),
        ))(input)
    }
}

#[derive(Debug)]
struct ResultType(Vec<ValType>);

impl ParseWasmBinary for ResultType {
    fn from_wasm_bytes(b: Span<'_>) -> nom::IResult<Span<'_>, Self> {
        use nom::combinator::map;
        map(Vec::<ValType>::from_wasm_bytes, ResultType)(b)
    }
}

#[derive(Debug)]
struct FuncType(ResultType, ResultType);

impl ParseWasmBinary for FuncType {
    fn from_wasm_bytes(b: Span<'_>) -> nom::IResult<Span<'_>, Self> {
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
            |(args, rets)| FuncType(args, rets),
        )(b)
    }
}

#[derive(Debug)]
enum Limits {
    Min(u32),
    Range(u32, u32),
}

impl ParseWasmBinary for Limits {
    fn from_wasm_bytes(b: Span<'_>) -> nom::IResult<Span<'_>, Self> {
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

#[derive(Debug)]
struct MemType(Limits);

impl ParseWasmBinary for MemType {
    fn from_wasm_bytes(b: Span<'_>) -> nom::IResult<Span<'_>, Self> {
        use nom::combinator::map;
        map(Limits::from_wasm_bytes, MemType)(b)
    }
}

#[derive(Debug)]
struct TableType(RefType, Limits);

impl ParseWasmBinary for TableType {
    fn from_wasm_bytes(b: Span<'_>) -> nom::IResult<Span<'_>, Self> {
        use nom::{combinator::map, sequence::tuple};
        map(
            tuple((RefType::from_wasm_bytes, Limits::from_wasm_bytes)),
            |(rt, limits)| TableType(rt, limits),
        )(b)
    }
}

#[derive(Debug)]
enum Mutability {
    Const,
    Variable,
}

impl ParseWasmBinary for Mutability {
    fn from_wasm_bytes(input: Span<'_>) -> nom::IResult<Span<'_>, Self> {
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

#[derive(Debug)]
struct GlobalType(ValType, Mutability);

impl ParseWasmBinary for GlobalType {
    fn from_wasm_bytes(b: Span<'_>) -> nom::IResult<Span<'_>, Self> {
        use nom::{combinator::map, sequence::tuple};
        map(
            tuple((ValType::from_wasm_bytes, Mutability::from_wasm_bytes)),
            |(ty, mutability)| GlobalType(ty, mutability),
        )(b)
    }
}

#[derive(Debug)]
enum BlockType {
    Empty,
    Val(ValType),
    TypeIndex(i32),
}

impl ParseWasmBinary for BlockType {
    fn from_wasm_bytes(b: Span<'_>) -> nom::IResult<Span<'_>, Self> {
        use nom::{branch::alt, bytes::complete::tag, combinator::map};

        alt((
            map(tag([0x40]), |_| BlockType::Empty),
            map(ValType::from_wasm_bytes, BlockType::Val),
            map(i64::from_wasm_bytes, |idx| {
                if !(0..=(1 << 33)).contains(&idx) {
                    todo!("we do not allow this index");
                }

                BlockType::TypeIndex(idx as i32)
            }),
        ))(b)
    }
}

#[derive(Debug)]
struct MemArg(u32, u32);

impl ParseWasmBinary for MemArg {
    fn from_wasm_bytes(input: Span<'_>) -> nom::IResult<Span<'_>, Self> {
        use nom::{combinator::map, sequence::tuple};

        map(
            tuple((u32::from_wasm_bytes, u32::from_wasm_bytes)),
            |(align, offset)| MemArg(align, offset),
        )(input)
    }
}

#[derive(Debug)]
enum Instr {
    // Control Instructions
    Unreachable,
    Nop,
    Block(BlockType, Vec<Instr>),
    Loop(BlockType, Vec<Instr>),
    If(BlockType, Vec<Instr>),
    IfElse(BlockType, Vec<Instr>, Vec<Instr>),
    Br(LabelIdx),
    BrIf(LabelIdx),
    BrTable(Vec<LabelIdx>, LabelIdx),
    Return,
    Call(FuncIdx),
    CallIndirect(TypeIdx, TableIdx),

    // Reference Instructions
    RefNull(RefType),
    RefIsNull,
    RefFunc(FuncIdx),

    // Parametric Instructions
    Drop,
    SelectEmpty,
    Select(Vec<ValType>),

    // Variable Instructions
    LocalGet(LocalIdx),
    LocalSet(LocalIdx),
    LocalTee(LocalIdx),
    GlobalGet(LocalIdx),
    GlobalSet(LocalIdx),

    // Table Instructions
    TableGet(TableIdx),
    TableSet(TableIdx),
    TableInit(ElemIdx, TableIdx),
    ElemDrop(ElemIdx),
    TableCopy(TableIdx, TableIdx),
    TableGrow(TableIdx),
    TableSize(TableIdx),
    TableFill(TableIdx),

    // Memory Instructions
    I32Load(MemArg),
    I64Load(MemArg),
    F32Load(MemArg),
    F64Load(MemArg),
    I32Load8S(MemArg),
    I32Load8U(MemArg),
    I32Load16S(MemArg),
    I32Load16U(MemArg),
    I64Load8S(MemArg),
    I64Load8U(MemArg),
    I64Load16S(MemArg),
    I64Load16U(MemArg),
    I64Load32S(MemArg),
    I64Load32U(MemArg),
    I32Store(MemArg),
    I64Store(MemArg),
    F32Store(MemArg),
    F64Store(MemArg),
    I32Store8(MemArg),
    I32Store16(MemArg),
    I64Store8(MemArg),
    I64Store16(MemArg),
    I64Store32(MemArg),
    MemorySize(MemIdx),
    MemoryGrow(MemIdx),
    MemoryInit(DataIdx, MemIdx),
    DataDrop(DataIdx),
    MemoryCopy(MemIdx, MemIdx),
    MemoryFill(MemIdx),

    // Numeric Instructions
    I32Const(i32),
    I64Const(i64),
    F32Const(f32),
    F64Const(f64),
    I32Eqz,
    I32Eq,
    I32Ne,
    I32LtS,
    I32LtU,
    I32GtS,
    I32GtU,
    I32LeS,
    I32LeU,
    I32GeS,
    I32GeU,
    I64Eqz,
    I64Eq,
    I64Ne,
    I64LtS,
    I64LtU,
    I64GtS,
    I64GtU,
    I64LeS,
    I64LeU,
    I64GeS,
    I64GeU,
    F32Eq,
    F32Ne,
    F32Lt,
    F32Gt,
    F32Le,
    F32Ge,
    F64Eq,
    F64Ne,
    F64Lt,
    F64Gt,
    F64Le,
    F64Ge,
    I32Clz,
    I32Ctz,
    I32Popcnt,
    I32DivS,
    I32DivU,
    I32RemS,
    I32RemU,
    I32And,
    I32Ior,
    I32Xor,
    I32Shl,
    I32ShrS,
    I32ShrU,
    I32Rol,
    I32Ror,
    I64Clz,
    I64Ctz,
    I64Popcnt,
    I64DivS,
    I64DivU,
    I64RemS,
    I64RemU,
    I64And,
    I64Ior,
    I64Xor,
    I64Shl,
    I64ShrS,
    I64ShrU,
    I64Rol,
    I64Ror,
    F32Abs,
    F32Neg,
    F32Ceil,
    F32Floor,
    F32Trunc,
    F32NearestInt,
    F32Sqrt,
    F32Add,
    F32Sub,
    F32Mul,
    F32Div,
    F32Min,
    F32Max,
    F32CopySign,
    F64Abs,
    F64Neg,
    F64Ceil,
    F64Floor,
    F64Trunc,
    F64NearestInt,
    F64Sqrt,
    F64Add,
    F64Sub,
    F64Mul,
    F64Div,
    F64Min,
    F64Max,
    F64CopySign,
    I32ConvertI64,
    I32SConvertF32,
    I32UConvertF32,
    I32SConvertF64,
    I32UConvertF64,
    I64SConvertI32,
    I64UConvertI32,
    I64SConvertF32,
    I64UConvertF32,
    I64SConvertF64,
    I64UConvertF64,
    F32SConvertI32,
    F32UConvertI32,
    F32SConvertI64,
    F32UConvertI64,
    F32ConvertF64,
    F64SConvertI32,
    F64UConvertI32,
    F64SConvertI64,
    F64UConvertI64,
    F64ConvertF32,
    I32ReinterpretF32,
    I64ReinterpretF64,
    F32ReinterpretI32,
    F64ReinterpretI64,
    I32SExtendI8,
    I32SExtendI16,
    I64SExtendI8,
    I64SExtendI16,
    I64SExtendI32,
    // Vector Instructions
}

fn control_instrs(b: Span<'_>) -> nom::IResult<Span<'_>, Instr> {
    use nom::{
        branch::alt,
        bytes::complete::tag,
        combinator::map,
        sequence::{delimited, preceded, tuple},
    };
    alt((
        map(tag([0x00]), |_| Instr::Unreachable),
        map(tag([0x01]), |_| Instr::Nop),
        map(
            delimited(
                tag([0x02]),
                tuple((BlockType::from_wasm_bytes, Vec::<Instr>::from_wasm_bytes)),
                tag([0x0b]),
            ),
            |(bt, ins)| Instr::Block(bt, ins),
        ),
        map(
            delimited(
                tag([0x03]),
                tuple((BlockType::from_wasm_bytes, Vec::<Instr>::from_wasm_bytes)),
                tag([0x0b]),
            ),
            |(bt, ins)| Instr::Loop(bt, ins),
        ),
        map(
            delimited(
                tag([0x04]),
                tuple((
                    BlockType::from_wasm_bytes,
                    Vec::<Instr>::from_wasm_bytes,
                    tag([0x05]),
                    Vec::<Instr>::from_wasm_bytes,
                )),
                tag([0x0b]),
            ),
            |(bt, consequent, _, alternate)| Instr::IfElse(bt, consequent, alternate),
        ),
        map(
            delimited(
                tag([0x04]),
                tuple((BlockType::from_wasm_bytes, Vec::<Instr>::from_wasm_bytes)),
                tag([0x0b]),
            ),
            |(bt, consequent)| Instr::If(bt, consequent),
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
            |(tbl, alt)| Instr::BrTable(tbl, alt),
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

fn ref_instrs(b: Span<'_>) -> nom::IResult<Span<'_>, Instr> {
    use nom::{branch::alt, bytes::complete::tag, combinator::map, sequence::preceded};
    alt((
        map(
            preceded(tag([0xd0]), RefType::from_wasm_bytes),
            Instr::RefNull,
        ),
        map(tag([0xd1]), |_| Instr::RefIsNull),
        map(
            preceded(tag([0xd0]), FuncIdx::from_wasm_bytes),
            Instr::RefFunc,
        ),
    ))(b)
}

fn parametric_instrs(b: Span<'_>) -> nom::IResult<Span<'_>, Instr> {
    use nom::{branch::alt, bytes::complete::tag, combinator::map, sequence::preceded};
    alt((
        map(tag([0x1a]), |_| Instr::Drop),
        map(tag([0x1b]), |_| Instr::SelectEmpty),
        map(
            preceded(tag([0x1c]), Vec::<ValType>::from_wasm_bytes),
            Instr::Select,
        ),
    ))(b)
}

fn variable_instrs(b: Span<'_>) -> nom::IResult<Span<'_>, Instr> {
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
            preceded(tag([0x23]), LocalIdx::from_wasm_bytes),
            Instr::GlobalGet,
        ),
        map(
            preceded(tag([0x24]), LocalIdx::from_wasm_bytes),
            Instr::GlobalSet,
        ),
    ))(b)
}

fn table_instrs(b: Span<'_>) -> nom::IResult<Span<'_>, Instr> {
    use nom::{
        branch::alt,
        bytes::complete::tag,
        combinator::map,
        sequence::{preceded, tuple},
    };

    alt((
        map(
            preceded(tag([0x25]), TableIdx::from_wasm_bytes),
            Instr::TableGet,
        ),
        map(
            preceded(tag([0x26]), TableIdx::from_wasm_bytes),
            Instr::TableGet,
        ),
        // TODO: these should be encoded as u32s, but instead I'm just... pulling
        // them in as byte literals, since they're all <2 bytes long in LEB128.
        map(
            preceded(
                tag([0xfc, 0x0c]),
                tuple((ElemIdx::from_wasm_bytes, TableIdx::from_wasm_bytes)),
            ),
            |(ei, ti)| Instr::TableInit(ei, ti),
        ),
        map(
            preceded(tag([0xfc, 0x0d]), ElemIdx::from_wasm_bytes),
            Instr::ElemDrop,
        ),
        map(
            preceded(
                tag([0xfc, 0x0e]),
                tuple((TableIdx::from_wasm_bytes, TableIdx::from_wasm_bytes)),
            ),
            |(ei, ti)| Instr::TableCopy(ei, ti),
        ),
        map(
            preceded(tag([0xfc, 0x0f]), TableIdx::from_wasm_bytes),
            Instr::TableGrow,
        ),
        map(
            preceded(tag([0xfc, 0x10]), TableIdx::from_wasm_bytes),
            Instr::TableSize,
        ),
        map(
            preceded(tag([0xfc, 0x11]), TableIdx::from_wasm_bytes),
            Instr::TableFill,
        ),
    ))(b)
}

fn memory_instrs(input: Span<'_>) -> nom::IResult<Span<'_>, Instr> {
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
        map(
            preceded(
                tag([0xfc, 0x08]),
                tuple((DataIdx::from_wasm_bytes, MemIdx::from_wasm_bytes)),
            ),
            |(di, mi)| Instr::MemoryInit(di, mi),
        ),
        map(
            preceded(tag([0xfc, 0x09]), DataIdx::from_wasm_bytes),
            Instr::DataDrop,
        ),
        map(
            preceded(
                tag([0xfc, 0x0a]),
                tuple((MemIdx::from_wasm_bytes, MemIdx::from_wasm_bytes)),
            ),
            |(mi0, mi1)| Instr::MemoryCopy(mi0, mi1),
        ),
        map(
            preceded(tag([0xfc, 0x0b]), MemIdx::from_wasm_bytes),
            Instr::MemoryFill,
        ),
    ))(input)
}

fn numeric_instrs(input: Span<'_>) -> nom::IResult<Span<'_>, Instr> {
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

impl ParseWasmBinary for Instr {
    fn from_wasm_bytes(b: Span<'_>) -> nom::IResult<Span<'_>, Self> {
        use nom::branch::alt;

        alt((
            control_instrs,
            ref_instrs,
            parametric_instrs,
            variable_instrs,
            table_instrs,
        ))(b)
    }
}

#[derive(Debug)]
struct TypeIdx(u32);
#[derive(Debug)]
struct FuncIdx(u32);
#[derive(Debug)]
struct TableIdx(u32);
#[derive(Debug)]
struct MemIdx(u32);
#[derive(Debug)]
struct GlobalIdx(u32);
#[derive(Debug)]
struct ElemIdx(u32);
#[derive(Debug)]
struct DataIdx(u32);
#[derive(Debug)]
struct LocalIdx(u32);
#[derive(Debug)]
struct LabelIdx(u32);

impl_parse_for_newtype!(TypeIdx, u32);
impl_parse_for_newtype!(FuncIdx, u32);
impl_parse_for_newtype!(TableIdx, u32);
impl_parse_for_newtype!(MemIdx, u32);
impl_parse_for_newtype!(GlobalIdx, u32);
impl_parse_for_newtype!(ElemIdx, u32);
impl_parse_for_newtype!(DataIdx, u32);
impl_parse_for_newtype!(LocalIdx, u32);
impl_parse_for_newtype!(LabelIdx, u32);

#[derive(Debug)]
enum Sections {
    Custom(TKTK),
    Type(TKTK),
    Import(TKTK),
    Function(TKTK),
    Table(TKTK),
    Memory(TKTK),
    Global(TKTK),
    Export(TKTK),
    Start(TKTK),
    Element(TKTK),
    Code(TKTK),
    Data(TKTK),
    DataCount(TKTK),
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
        let (rest, v) = u32::from_wasm_bytes(Span::new(&v[..])).unwrap();
        assert_eq!(v, 8082008);
        assert_eq!(&rest[..], b"");

        let mut v = Vec::new();
        leb128::write::unsigned(&mut v, u32::MAX as u64).unwrap();
        let (rest, v) = u32::from_wasm_bytes(Span::new(&v[..])).unwrap();
        assert_eq!(v, u32::MAX);
        assert_eq!(&rest[..], b"");

        let mut v = Vec::new();
        leb128::write::unsigned(&mut v, u32::MIN as u64).unwrap();
        let (rest, v) = u32::from_wasm_bytes(Span::new(&v[..])).unwrap();
        assert_eq!(v, u32::MIN);
        assert_eq!(&rest[..], b"");

        let mut v = Vec::new();
        leb128::write::unsigned(&mut v, 0).unwrap();
        let (rest, v) = u32::from_wasm_bytes(Span::new(&v[..])).unwrap();
        assert_eq!(v, 0);
        assert_eq!(&rest[..], b"");
    }

    #[test]
    fn test_read_i32() {
        use leb128;
        let mut v = Vec::new();
        leb128::write::signed(&mut v, 8082008).unwrap();
        let (rest, v) = i32::from_wasm_bytes(Span::new(&v[..])).unwrap();
        assert_eq!(v, 8082008);
        assert_eq!(&rest[..], b"");

        let mut v = Vec::new();
        leb128::write::signed(&mut v, -45).unwrap();
        let (rest, v) = i32::from_wasm_bytes(Span::new(&v[..])).unwrap();
        assert_eq!(v, -45);
        assert_eq!(&rest[..], b"");

        let mut v = Vec::new();
        leb128::write::signed(&mut v, i32::MAX as i64).unwrap();
        let (rest, v) = i32::from_wasm_bytes(Span::new(&v[..])).unwrap();
        assert_eq!(v, i32::MAX);
        assert_eq!(&rest[..], b"");

        let mut v = Vec::new();
        leb128::write::signed(&mut v, i32::MIN as i64).unwrap();
        let (rest, v) = i32::from_wasm_bytes(Span::new(&v[..])).unwrap();
        assert_eq!(v, i32::MIN);
        assert_eq!(&rest[..], b"");

        let mut v = Vec::new();
        leb128::write::signed(&mut v, 0).unwrap();
        let (rest, v) = i32::from_wasm_bytes(Span::new(&v[..])).unwrap();
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
        let (rest, v) = Vec::<i64>::from_wasm_bytes(Span::new(&v[..])).unwrap();
        assert_eq!(v, vec![1, 1, 2]);
        assert_eq!(&rest[..], [3, 5]);
    }

    #[test]
    fn test_read_string() {
        use leb128;
        let xs = "hello world!";
        let mut v = Vec::new();
        leb128::write::unsigned(&mut v, xs.len() as u64).unwrap();
        write!(&mut v, "{}", xs).unwrap();

        let (rest, v) = String::from_wasm_bytes(Span::new(&v[..])).unwrap();
        assert_eq!(v, "hello world!");
        assert_eq!(&rest[..], []);
    }
}
