#![allow(dead_code)]

use std::io::{Cursor, Seek};

use nom_locate::LocatedSpan;
struct LabelIdx;
struct FuncIdx;
struct Blocktype;
struct TypeIdx;
struct TableIdx;

type Span<'a> = LocatedSpan<&'a [u8]>;

// A TODO struct that borrows from journalism [1]
// [1]: https://en.wikipedia.org/wiki/To_come_(publishing)
#[allow(clippy::upper_case_acronyms)]
struct TKTK;

trait FromWasmBytes: Sized {
    fn from_wasm_bytes(b: Span<'_>) -> nom::IResult<Span<'_>, Self>;
}

impl<T: FromWasmBytes> FromWasmBytes for Vec<T> {
    fn from_wasm_bytes(b: Span<'_>) -> nom::IResult<Span<'_>, Self> {
        let (mut rest, sz) = <u32 as FromWasmBytes>::from_wasm_bytes(b)?;
        let mut v = Vec::with_capacity(sz as usize);
        for _ in 0..sz {
            let (r, item) = <T as FromWasmBytes>::from_wasm_bytes(rest)?;
            v.push(item);
            rest = r;
        }
        Ok((rest, v))
    }
}

macro_rules! parse_leb128 {
    ($type:ident, $mode:ident) => {
        impl FromWasmBytes for $type {
            fn from_wasm_bytes(b: Span<'_>) -> nom::IResult<Span<'_>, Self> {
                use nom::bytes::complete::take;
                let mut bytes = Cursor::new(&b[..]);
                let value = match leb128::read::$mode(&mut bytes) {
                    Ok(v) => v,
                    Err(leb128::read::Error::IoError(e)) => match e.kind() {
                        std::io::ErrorKind::UnexpectedEof => todo!(),
                        _ => todo!()
                    },
                    _err => return todo!()
                };
                let (rest, _) = take(bytes.stream_position().unwrap() as usize)(b)?;

                Ok((rest, value as $type))
            }
        }

    };
}

parse_leb128!(i32, signed);
parse_leb128!(i64, signed);
parse_leb128!(u32, unsigned);
parse_leb128!(u64, unsigned);


enum Section {
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

enum Instr {
    Unreachable,
    Nop,
    Block(Blocktype, Vec<Instr>),
    Loop(Blocktype, Vec<Instr>),
    If(Blocktype, Vec<Instr>),
    IfElse(Blocktype, Vec<Instr>),
    Br(LabelIdx),
    BrIf(LabelIdx),
    BrTable(Vec<LabelIdx>, LabelIdx),
    Return,
    Call(FuncIdx),
    CallIndirect(TypeIdx, TableIdx),
}
