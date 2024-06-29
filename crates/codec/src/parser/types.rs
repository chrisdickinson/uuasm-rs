use std::mem;

use uuasm_nodes::{NumType, RefType, ResultType, Type, ValType, VecType};

use crate::{window::DecodeWindow, Advancement, Parse, ParseError, ParseResult};

use super::{accumulator::Accumulator, leb::LEBParser, state::ParseState, take::Take};

pub(crate) enum TypeParser {
    Init(Vec<Type>),
    InputSize(Vec<Type>, u32),
    Input(Vec<Type>, Option<ResultType>),
    OutputSize(Vec<Type>, Option<ResultType>, u32),
    Output(Vec<Type>, Option<ResultType>, Option<ResultType>),
}

impl Default for TypeParser {
    fn default() -> Self {
        Self::Init(vec![])
    }
}

impl TypeParser {
    fn map_buffer_to_result_type(input_buf: &[u8]) -> Result<ResultType, ParseError> {
        let mut types = Vec::with_capacity(input_buf.len());
        for item in input_buf {
            types.push(match item {
                0x6f => ValType::RefType(RefType::ExternRef),
                0x70 => ValType::RefType(RefType::FuncRef),
                0x7b => ValType::VecType(VecType::V128),
                0x7c => ValType::NumType(NumType::F64),
                0x7d => ValType::NumType(NumType::F32),
                0x7e => ValType::NumType(NumType::I64),
                0x7f => ValType::NumType(NumType::I32),
                byte => return Err(ParseError::BadType(*byte)),
            })
        }
        Ok(ResultType(types.into()))
    }
}

impl Parse for TypeParser {
    type Production = Box<[Type]>;

    fn advance(&mut self, mut window: DecodeWindow) -> ParseResult {
        loop {
            *self = match self {
                TypeParser::Init(_) => {
                    match window.peek() {
                        Err(ParseError::Expected(1)) => {
                            return Ok(Advancement::Ready(window.offset()));
                        }
                        Err(err) => return Err(err),
                        _ => {}
                    }

                    let tag = window.take()?;
                    if tag != 0x60 {
                        return Err(ParseError::BadTypePrefix(tag));
                    }

                    return Ok(Advancement::YieldTo(
                        window.offset(),
                        ParseState::LEBU32(LEBParser::default()),
                        |last_state, this_state| {
                            let ParseState::LEBU32(leb) = last_state else {
                                unreachable!();
                            };

                            let ParseState::TypeSection(take) = this_state else {
                                unreachable!();
                            };

                            let entry_count = leb.production()?;
                            Ok(ParseState::TypeSection(take.map(|this_state| {
                                let Self::Init(v) = this_state else {
                                    unreachable!();
                                };
                                Ok(if entry_count == 0 {
                                    Self::Input(v, None)
                                } else {
                                    Self::InputSize(v, entry_count)
                                })
                            })?))
                        },
                    ));
                }
                TypeParser::InputSize(_, size) => {
                    let size = *size as usize;

                    return Ok(Advancement::YieldTo(
                        window.offset(),
                        ParseState::Accumulate(Take::new(Accumulator::new(size), size)),
                        |last_state, this_state| {
                            let ParseState::Accumulate(accum) = last_state else {
                                unreachable!()
                            };
                            let ParseState::TypeSection(take) = this_state else {
                                unreachable!();
                            };

                            let input_buf = accum.production()?;
                            let result_type = TypeParser::map_buffer_to_result_type(&input_buf)?;
                            Ok(ParseState::TypeSection(take.map(|this_state| {
                                let Self::InputSize(v, _) = this_state else {
                                    unreachable!();
                                };
                                Ok(Self::Input(v, Some(result_type)))
                            })?))
                        },
                    ));
                }

                TypeParser::Input(_, _) => {
                    return Ok(Advancement::YieldTo(
                        window.offset(),
                        ParseState::LEBU32(LEBParser::default()),
                        |last_state, this_state| {
                            let ParseState::LEBU32(leb) = last_state else {
                                unreachable!();
                            };

                            let ParseState::TypeSection(take) = this_state else {
                                unreachable!();
                            };

                            let entry_count = leb.production()?;

                            Ok(ParseState::TypeSection(take.map(|this_state| {
                                let Self::Input(v, result_type) = this_state else {
                                    unreachable!();
                                };
                                Ok(if entry_count == 0 {
                                    Self::Output(v, result_type, None)
                                } else {
                                    Self::OutputSize(v, result_type, entry_count)
                                })
                            })?))
                        },
                    ));
                }
                TypeParser::OutputSize(_, _, size) => {
                    let size = *size as usize;

                    return Ok(Advancement::YieldTo(
                        window.offset(),
                        ParseState::Accumulate(Take::new(Accumulator::new(size), size)),
                        |last_state, this_state| {
                            let ParseState::Accumulate(accum) = last_state else {
                                unreachable!()
                            };
                            let ParseState::TypeSection(take) = this_state else {
                                unreachable!();
                            };

                            let output_buf = accum.production()?;
                            let result_type = TypeParser::map_buffer_to_result_type(&output_buf)?;

                            Ok(ParseState::TypeSection(take.map(|this_state| {
                                let Self::OutputSize(v, input_result_type, _) = this_state else {
                                    unreachable!();
                                };
                                Ok(Self::Output(v, input_result_type, Some(result_type)))
                            })?))
                        },
                    ));
                }
                TypeParser::Output(v, input_type, output_type) => {
                    v.push(Type(
                        input_type.take().unwrap_or_default(),
                        output_type.take().unwrap_or_default(),
                    ));
                    TypeParser::Init(mem::take(v))
                }
            }
        }
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        let Self::Init(v) = self else {
            unreachable!();
        };

        Ok(v.into_boxed_slice())
    }
}
