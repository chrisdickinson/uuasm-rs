use uuasm_ir::IR;

use crate::{
    parser::instrarg_multibyte::InstrArgMultibyteParser, window::DecodeWindow, Advancement,
    IRError, Parse, ParseErrorKind,
};

use super::{accumulator::Accumulator, any::AnyParser, block::BlockParser, repeated::Repeated};

#[derive(Default)]
enum State {
    #[default]
    Empty,
    LastInstr(u8),
}

pub struct ExprParser<T: IR> {
    instrs: Vec<T::Instr>,
    state: State,
    shift_last: bool,
}

impl<T: IR> ExprParser<T> {
    pub(crate) fn no_shift() -> Self {
        Self {
            instrs: Vec::new(),
            state: State::default(),
            shift_last: false,
        }
    }
}

impl<T: IR> Default for ExprParser<T> {
    fn default() -> Self {
        Self {
            instrs: Vec::new(),
            state: State::default(),
            shift_last: true,
        }
    }
}

impl<T: IR> Parse<T> for ExprParser<T> {
    type Production = T::Expr;

    fn advance(&mut self, irgen: &mut T, window: &mut DecodeWindow) -> crate::ParseResult<T> {
        loop {
            let next = window.peek()?;
            #[allow(clippy::manual_range_patterns)]
            match next {
                // unused instrs
                0x27 |
                0xd3..=0xfb |
                0xc5..=0xcf |
                0x1d..=0x1f |
                0x12..=0x19 |
                0x06..=0x0a => return Err(ParseErrorKind::BadInstruction(next)),

                // else
                0x05 |
                // end block
                0x0b => {
                    if self.shift_last {
                        let _ = window.take();
                    }

                    return Ok(Advancement::Ready)
                },

                // ## wingding: vec of u32 values, followed by a u32 value
                // - br.table
                0x0e => {
                    let _ = window.take();
                    return Ok(Advancement::YieldTo( AnyParser::ArgTable(Default::default()), |irgen, last_state, this_state| {
                        let AnyParser::ArgTable(parser) = last_state else {  unsafe { crate::cold(); std::hint::unreachable_unchecked() } };
                        let AnyParser::Expr(Self { mut instrs, shift_last, .. }) = this_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let (items, alternate) = parser.production(irgen)?;
                        irgen.make_instr_table(&items, alternate, &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs, shift_last }))
                    }))
                }

                // ## blocktype -> nesting
                // - block
                0x02 |
                // - loop
                0x03 => {
                    let _ = window.take();
                    self.state = State::LastInstr(next);
                    return Ok(Advancement::YieldTo( AnyParser::Block(BlockParser::Init(next)), |irgen, last_state, this_state| {
                        let AnyParser::Block(parser) = last_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let AnyParser::Expr(Self { state: State::LastInstr(code), mut instrs, shift_last }) = this_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let (block_type, block_instrs) = parser.production(irgen)?;
                        irgen.make_instr_block(code, block_type, block_instrs, &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs, shift_last }))
                    }))
                }

                // - if
                0x04 => {
                    let _ = window.take();
                    return Ok(Advancement::YieldTo( AnyParser::IfElseBlock(Default::default()), |irgen, last_state, this_state| {
                        let AnyParser::IfElseBlock(parser) = last_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let AnyParser::Expr(Self { mut instrs, shift_last, .. }) = this_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let (block_type, consequent, alternate) = parser.production(irgen)?;
                        irgen.make_instr_block_ifelse(block_type, consequent, alternate, &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs, shift_last }))
                    }))
                },


                // ## wingding: single 8 bit value
                // - ref.null
                0xd0 => {
                    let _ = window.take();
                    return Ok(Advancement::YieldTo( AnyParser::ArgRefNull(Default::default()), |irgen, last_state, this_state| {
                        let AnyParser::ArgRefNull(parser) = last_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let AnyParser::Expr(Self { mut instrs, shift_last, .. }) = this_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let value = parser.production(irgen)?;
                        irgen.make_instr_arity1(0xd0, 0, value as u32, &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs, shift_last }))
                    }))
                },
                // ## wingding: vec of valtype (8-bit values)
                // - select
                0x1c => {
                    let _ = window.take();
                    return Ok(Advancement::YieldTo( AnyParser::ByteVec(Default::default()), |irgen, last_state, this_state| {
                        let AnyParser::ByteVec(parser) = last_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let AnyParser::Expr(Self { mut instrs, shift_last, .. }) = this_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let types = parser.production(irgen)?;
                        let types: Box<[T::ValType]> = types
                            .iter()
                            .map(|xs| irgen.make_val_type(*xs))
                            .collect::<Result<_, T::Error>>()
                            .map_err(IRError)?;
                        irgen.make_instr_select(types, &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs, shift_last }))
                    }))
                }

                // ## single 64-bit args
                // - f64 const
                0x44 => {
                    let _ = window.take();
                    self.state = State::LastInstr(next);
                    return Ok(Advancement::YieldTo( AnyParser::Accumulate(Accumulator::new(8)), |irgen, last_state, this_state| {
                        let AnyParser::Accumulate(parser) = last_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let AnyParser::Expr(Self { state: State::LastInstr(code), mut instrs, shift_last }) = this_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let lhs = parser.production(irgen)?;
                        if lhs.len() != 8 {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        }
                        let lhs: u64 = u64::from_le_bytes((&*lhs).try_into().unwrap());
                        irgen.make_instr_arity1_64(code, 0, lhs, &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs, shift_last }))
                    }))
                },

                // ## single 32-bit args
                // - f32 const
                0x43 => {
                    let _ = window.take();
                    self.state = State::LastInstr(next);
                    return Ok(Advancement::YieldTo( AnyParser::Accumulate(Accumulator::new(4)), |irgen, last_state, this_state| {
                        let AnyParser::Accumulate(parser) = last_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let AnyParser::Expr(Self { state: State::LastInstr(code), mut instrs, shift_last }) = this_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let lhs = parser.production(irgen)?;
                        if lhs.len() != 4 {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        }
                        let lhs: u32 = u32::from_le_bytes((&*lhs).try_into().unwrap());
                        irgen.make_instr_arity1(code, 0, lhs, &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs, shift_last }))
                    }))
                },

                // ## single leb 64-bit args
                // - i64 const
                0x42 => {
                    let _ = window.take();
                    self.state = State::LastInstr(next);
                    return Ok(Advancement::YieldTo( AnyParser::LEBI64(Default::default()), |irgen, last_state, this_state| {
                        let AnyParser::LEBI64(parser) = last_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let AnyParser::Expr(Self { state: State::LastInstr(code), mut instrs, shift_last }) = this_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let lhs = parser.production(irgen)?;
                        irgen.make_instr_arity1_64(code, 0, lhs as u64, &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs, shift_last }))
                    }))
                }
,

                // ## two leb 32-bit args
                // - call.indirect
                0x11 |
                // - memory instrs; memidx
                0x28..=0x3e => {
                    let _ = window.take();
                    self.state = State::LastInstr(next);
                    return Ok(Advancement::YieldTo( AnyParser::RepeatedLEBU32(Repeated::times(2)), |irgen, last_state, this_state| {
                        let AnyParser::RepeatedLEBU32(parser) = last_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let AnyParser::Expr(Self { state: State::LastInstr(code), mut instrs, shift_last }) = this_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let items = parser.production(irgen)?;
                        irgen.make_instr_arity2(code, 0, items[0], items[1], &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs, shift_last }))
                    }))
                }

                // - i32 const (we specifically need an i32 value)
                0x41 => {
                    let _ = window.take();
                    self.state = State::LastInstr(next);
                    return Ok(Advancement::YieldTo( AnyParser::LEBI32(Default::default()), |irgen, last_state, this_state| {
                        let AnyParser::LEBI32(parser) = last_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let AnyParser::Expr(Self { state: State::LastInstr(code), mut instrs, shift_last, }) = this_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let arg0 = parser.production(irgen)?;
                        irgen.make_instr_arity1(code, 0, arg0 as u32, &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs, shift_last, }))
                    }))
                }

                // ## single leb 32-bit args
                // - func.call
                0x10 |
                // - variables ({local,global}.{get,set.tee}), one index
                0x20..=0x24 |
                // - table get/set
                0x25 | 0x26 |
                // - memory size/grow
                0x3f | 0x40 |
                // - ref.func
                0xd2 |
                // - br, br.if
                0x0c | 0x0d => {
                    let _ = window.take();
                    self.state = State::LastInstr(next);
                    return Ok(Advancement::YieldTo( AnyParser::LEBU32(Default::default()), |irgen, last_state, this_state| {
                        let AnyParser::LEBU32(parser) = last_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let AnyParser::Expr(Self { state: State::LastInstr(code), mut instrs, shift_last, }) = this_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let arg0 = parser.production(irgen)?;
                        irgen.make_instr_arity1(code, 0, arg0, &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs, shift_last, }))
                    }))
                }

                // ## no args
                // - unreachable, nop
                0x00 | 0x01 |
                // - ref.is_null
                0xd1 |
                // - drop and select empty
                0x1a | 0x1b |
                // - return
                0x0f |
                // - numeric instrs
                0x45..=0xc4 => {
                    let _ = window.take();
                    irgen.make_instr_arity0(next, 0, &mut self.instrs).map_err(IRError)?;
                },

                // multibyte instrs
                0xfc | 0xfd | 0xfe | 0xff => {
                    let _ = window.take();
                    self.state = State::LastInstr(next);
                    return Ok(Advancement::YieldTo(AnyParser::ArgMultibyte(next.into()), |irgen, last_state, this_state| {
                        let AnyParser::ArgMultibyte(parser) = last_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };
                        let AnyParser::Expr(Self { state: State::LastInstr(code), mut instrs, shift_last, }) = this_state else {
                             unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                        };


                        match parser {
                            InstrArgMultibyteParser::Ident(leader, ident) => irgen.make_instr_arity0(leader, ident, &mut instrs),
                            InstrArgMultibyteParser::IdentArity1(leader, ident, arg0) => irgen.make_instr_arity1(leader, ident, arg0, &mut instrs),
                            InstrArgMultibyteParser::IdentArity2(leader, ident, arg0, arg1) => irgen.make_instr_arity2(leader, ident, arg0, arg1, &mut instrs),
                            InstrArgMultibyteParser::IdentArity3(_leader, _ident, _arg0, _arg1, _arg2) => todo!(),
                            InstrArgMultibyteParser::IdentArity4(_leader, _ident, _arg0, _arg1, _arg2, _arg3) => todo!(),
                            _ => return Err(ParseErrorKind::BadInstruction(code))
                        }.map_err(IRError)?;

                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs, shift_last, }))
                    }))

                },
            }
        }
    }

    fn production(
        self,
        irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseErrorKind<<T as IR>::Error>> {
        let Self { instrs, .. } = self;

        Ok(irgen.make_expr(instrs).map_err(IRError)?)
    }
}
