use uuasm_nodes::IR;

use crate::{Advancement, Parse};

use super::any::AnyParser;

pub struct InstrParser<T: IR>(Option<T::Export>);

impl<T: IR> Default for InstrParser<T> {
    fn default() -> Self {
        Self(None)
    }
}

/*
    // instr encoding:
    [0x00, 0xC4] -- single byte instrs
    [0xfc, 0xff] -- multi byte instrs

    // dingbats
    TypeInstrs(BlockType, Box<[Instr]>),
    TypeInstrs2(BlockType, Box<[Instr]>, Box<[Instr]>),
    BrTable(Box<[LabelIdx]>, LabelIdx),
    Select(Box<[ValType]>),

    // 1 u64
    F64(f64),
    I64(i64),

    // 2 u32
    MemDataOp(DataIdx, MemIdx),
    TableElemOp(ElemIdx, TableIdx),
    TableOp2(TableIdx, TableIdx),
    TypeTableOp(TypeIdx, TableIdx),
    MemOp2(MemIdx, MemIdx),
    LoadOp(MemArg),

    // 1 u32
    TableOp(TableIdx),
    MemOp(MemIdx),
    F32(f32),
    I32(i32),
    FuncOp(FuncIdx),
    GlobalOp(GlobalIdx),
    LabelOp(LabelIdx),
    LocalOp(LocalIdx),
    ElemOp(ElemIdx),
    DataOp(DataIdx),

    // 1 u8
    RefOp(RefType),

    // -----

    TypeInstrs(BlockType, Box<[Instr]>),
    TypeInstrs2(BlockType, Box<[Instr]>, Box<[Instr]>),
    BrTable(Box<[LabelIdx]>, LabelIdx),
    Select(Box<[ValType]>),
    DataOp(DataIdx),
    MemDataOp(DataIdx, MemIdx),
    ElemOp(ElemIdx),
    TableElemOp(ElemIdx, TableIdx),
    FuncOp(FuncIdx),
    GlobalOp(GlobalIdx),
    LabelOp(LabelIdx),
    LocalOp(LocalIdx),
    LoadOp(MemArg),
    MemOp(MemIdx),
    MemOp2(MemIdx, MemIdx),
    RefOp(RefType),
    TableOp(TableIdx),
    TableOp2(TableIdx, TableIdx),
    TypeTableOp(TypeIdx, TableIdx),
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
*/

impl<T: IR> Parse<T> for InstrParser<T> {
    type Production = T::Instr;

    fn advance(
        &mut self,
        _irgen: &mut T,
        window: crate::window::DecodeWindow,
    ) -> crate::ParseResult<T> {
        // XXX(chrisdickinson, 2024-07-09): I'm going to fold instr parsing up into expr parsing.
        if self.0.is_some() {
            return Ok(Advancement::Ready(window.offset()));
        }
        Ok(Advancement::YieldTo(
            window.offset(),
            AnyParser::LEBU32(Default::default()),
            |_irgen, last_state, _| {
                let AnyParser::LEBU32(_parser) = last_state else {
                    unreachable!();
                };

                todo!()

                // Ok(AnyParser::Instr(Self(Some(idx))))
            },
        ))
    }

    fn production(
        self,
        _irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseError<<T as IR>::Error>> {
        let Self(Some(_production)) = self else {
            unreachable!()
        };

        todo!()
    }
}
