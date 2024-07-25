use uuasm_nodes::IR;

use crate::{Advancement, IRError, Parse, ParseError};

use super::{accumulator::Accumulator, any::AnyParser, repeated::Repeated};

#[derive(Default)]
enum State {
    #[default]
    Empty,
    LastInstr(u8),
}

pub struct ExprParser<T: IR> {
    instrs: Vec<T::Instr>,
    state: State,
}

impl<T: IR> Default for ExprParser<T> {
    fn default() -> Self {
        Self {
            instrs: Vec::new(),
            state: State::default(),
        }
    }
}

impl<T: IR> Parse<T> for ExprParser<T> {
    type Production = T::Expr;

    fn advance(
        &mut self,
        irgen: &mut T,
        mut window: crate::window::DecodeWindow,
    ) -> crate::ParseResult<T> {
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
                0x06..=0x0a => return Err(ParseError::BadInstruction(next)),

                // else
                0x05 |
                // end block
                0x0b => return Ok(Advancement::Ready(window.offset())),

                // ## wingding: vec of u32 values, followed by a u32 value
                // - br.table
                0x0e => {
                    window.take().unwrap();
                    return Ok(Advancement::YieldTo(window.offset(), AnyParser::ArgTable(Default::default()), |irgen, last_state, this_state| {
                        let AnyParser::ArgTable(parser) = last_state else { unreachable!() };
                        let AnyParser::Expr(Self { mut instrs, .. }) = this_state else {
                            unreachable!();
                        };
                        let (items, alternate) = parser.production(irgen)?;
                        irgen.make_instr_table(items, alternate, &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs }))
                    }))
                }

                // ## blocktype -> nesting
                // - block
                0x02 |
                // - loop
                0x03 => {
                    window.take().unwrap();
                    self.state = State::LastInstr(next);
                    return Ok(Advancement::YieldTo(window.offset(), AnyParser::Block(Default::default()), |irgen, last_state, this_state| {
                        let AnyParser::Block(parser) = last_state else {
                            unreachable!();
                        };
                        let AnyParser::Expr(Self { state: State::LastInstr(code), mut instrs }) = this_state else {
                            unreachable!();
                        };
                        let (block_type, block_instrs) = parser.production(irgen)?;
                        irgen.make_instr_block(code, block_type, block_instrs, &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs }))
                    }))
                }

                // - if
                0x04 => {
                    window.take().unwrap();
                    return Ok(Advancement::YieldTo(window.offset(), AnyParser::IfElseBlock(Default::default()), |irgen, last_state, this_state| {
                        let AnyParser::IfElseBlock(parser) = last_state else {
                            unreachable!();
                        };
                        let AnyParser::Expr(Self { mut instrs, .. }) = this_state else {
                            unreachable!();
                        };
                        let (block_type, consequent, alternate) = parser.production(irgen)?;
                        irgen.make_instr_block_ifelse(block_type, consequent, alternate, &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs }))
                    }))
                },


                // ## wingding: single 8 bit value
                // - ref.null
                0xd0 => {
                    window.take().unwrap();
                    return Ok(Advancement::YieldTo(window.offset(), AnyParser::ArgRefNull(Default::default()), |irgen, last_state, this_state| {
                        let AnyParser::ArgRefNull(parser) = last_state else {
                            unreachable!();
                        };
                        let AnyParser::Expr(Self { mut instrs, .. }) = this_state else {
                            unreachable!();
                        };
                        let value = parser.production(irgen)?;
                        irgen.make_instr_unary(0xd0, value as u32, &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs }))
                    }))
                },

                // ## wingding: vec of valtype (8-bit values)
                // - select
                0x1c => {
                    window.take().unwrap();
                    return Ok(Advancement::YieldTo(window.offset(), AnyParser::ByteVec(Default::default()), |irgen, last_state, this_state| {
                        let AnyParser::ByteVec(parser) = last_state else {
                            unreachable!();
                        };
                        let AnyParser::Expr(Self { mut instrs, .. }) = this_state else {
                            unreachable!();
                        };
                        let types = parser.production(irgen)?;
                        let types: Box<[T::ValType]> = types
                            .iter()
                            .map(|xs| irgen.make_val_type(*xs))
                            .collect::<Result<_, T::Error>>()
                            .map_err(IRError)?;
                        irgen.make_instr_select(types, &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs }))
                    }))
                }

                // ## single 64-bit args
                // - f64 const
                0x44 => {
                    window.take().unwrap();
                    self.state = State::LastInstr(next);
                    return Ok(Advancement::YieldTo(window.offset(), AnyParser::Accumulate(Accumulator::new(8)), |irgen, last_state, this_state| {
                        let AnyParser::Accumulate(parser) = last_state else {
                            unreachable!();
                        };
                        let AnyParser::Expr(Self { state: State::LastInstr(code), mut instrs }) = this_state else {
                            unreachable!();
                        };
                        let lhs = parser.production(irgen)?;
                        if lhs.len() != 8 {
                            unreachable!();
                        }
                        let lhs: u64 = u64::from_le_bytes((&*lhs).try_into().unwrap());
                        irgen.make_instr_unary64(code, lhs, &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs }))
                    }))
                },

                // ## single 32-bit args
                // - f32 const
                0x43 => {
                    window.take().unwrap();
                    self.state = State::LastInstr(next);
                    return Ok(Advancement::YieldTo(window.offset(), AnyParser::Accumulate(Accumulator::new(4)), |irgen, last_state, this_state| {
                        let AnyParser::Accumulate(parser) = last_state else {
                            unreachable!();
                        };
                        let AnyParser::Expr(Self { state: State::LastInstr(code), mut instrs }) = this_state else {
                            unreachable!();
                        };
                        let lhs = parser.production(irgen)?;
                        if lhs.len() != 4 {
                            unreachable!();
                        }
                        let lhs: u32 = u32::from_le_bytes((&*lhs).try_into().unwrap());
                        irgen.make_instr_unary(code, lhs, &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs }))
                    }))
                },

                // ## single leb 64-bit args
                // - i64 const
                0x42 => {
                    window.take().unwrap();
                    self.state = State::LastInstr(next);
                    return Ok(Advancement::YieldTo(window.offset(), AnyParser::LEBU64(Default::default()), |irgen, last_state, this_state| {
                        let AnyParser::LEBU64(parser) = last_state else {
                            unreachable!();
                        };
                        let AnyParser::Expr(Self { state: State::LastInstr(code), mut instrs }) = this_state else {
                            unreachable!();
                        };
                        let lhs = parser.production(irgen)?;
                        irgen.make_instr_unary64(code, lhs, &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs }))
                    }))
                }
,

                // ## two leb 32-bit args
                // - call.indirect
                0x11 |
                // - memory instrs; memidx
                0x28..=0x3e => {
                    window.take().unwrap();
                    self.state = State::LastInstr(next);
                    return Ok(Advancement::YieldTo(window.offset(), AnyParser::RepeatedLEBU32(Repeated::times(2)), |irgen, last_state, this_state| {
                        let AnyParser::RepeatedLEBU32(parser) = last_state else {
                            unreachable!();
                        };
                        let AnyParser::Expr(Self { state: State::LastInstr(code), mut instrs }) = this_state else {
                            unreachable!();
                        };
                        let items = parser.production(irgen)?;
                        irgen.make_instr_binary(code, items[0], items[1], &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs }))
                    }))
                }

                // ## single leb 32-bit args
                // - func.call
                0x10 |
                // - i32 const
                0x41 |
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
                    window.take().unwrap();
                    self.state = State::LastInstr(next);
                    return Ok(Advancement::YieldTo(window.offset(), AnyParser::LEBU32(Default::default()), |irgen, last_state, this_state| {
                        let AnyParser::LEBU32(parser) = last_state else {
                            unreachable!();
                        };
                        let AnyParser::Expr(Self { state: State::LastInstr(code), mut instrs }) = this_state else {
                            unreachable!();
                        };
                        let arg0 = parser.production(irgen)?;
                        irgen.make_instr_unary(code, arg0, &mut instrs).map_err(IRError)?;
                        Ok(AnyParser::Expr(Self { state: State::Empty, instrs }))
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
                    window.take().unwrap();
                    irgen.make_instr_nullary(next, &mut self.instrs).map_err(IRError)?;
                },

                // multibyte instrs
                0xfc | 0xfd | 0xfe | 0xff => todo!(),
            }
        }
    }

    fn production(
        self,
        irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseError<<T as IR>::Error>> {
        let Self { instrs, .. } = self;

        Ok(irgen.make_expr(instrs).map_err(IRError)?)
    }
}
