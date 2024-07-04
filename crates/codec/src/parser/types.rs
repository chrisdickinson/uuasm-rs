use uuasm_nodes::IR;

use crate::{window::DecodeWindow, Advancement, Parse, ParseError, ParseResult};

use super::{accumulator::Accumulator, any::AnyParser, leb::LEBParser};

#[derive(Default)]
pub enum TypeParser<T: IR> {
    #[default]
    Init,
    InputSize(u32),
    Input(Option<T::ResultType>),
    OutputSize(Option<T::ResultType>, u32),
    Output(Option<T::ResultType>, Option<T::ResultType>),
}

impl<T: IR> TypeParser<T> {
    fn map_buffer_to_result_type(_input_buf: &[u8]) -> Result<T::ResultType, ParseError> {
        todo!()
        /*
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
        */
    }
}

impl<T: IR> Parse<T> for TypeParser<T> {
    type Production = <T as IR>::Type;

    fn advance(&mut self, _irgen: &mut T, mut window: DecodeWindow) -> ParseResult<T> {
        match self {
            TypeParser::Init => {
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

                Ok(Advancement::YieldTo(
                    window.offset(),
                    AnyParser::LEBU32(LEBParser::default()),
                    |irgen, last_state, this_state| {
                        let AnyParser::LEBU32(leb) = last_state else {
                            unreachable!();
                        };

                        let AnyParser::Type(Self::Init) = this_state else {
                            unreachable!();
                        };

                        let entry_count = leb.production(irgen)?;
                        Ok(AnyParser::Type(if entry_count == 0 {
                            Self::Input(None)
                        } else {
                            Self::InputSize(entry_count)
                        }))
                    },
                ))
            }
            TypeParser::InputSize(size) => {
                let size = *size as usize;

                Ok(Advancement::YieldTo(
                    window.offset(),
                    AnyParser::Accumulate(Accumulator::new(size)),
                    |irgen, last_state, this_state| {
                        let AnyParser::Accumulate(accum) = last_state else {
                            unreachable!()
                        };
                        let AnyParser::Type(Self::InputSize(_)) = this_state else {
                            unreachable!();
                        };

                        let input_buf = accum.production(irgen)?;
                        let result_type = Self::map_buffer_to_result_type(&input_buf)?;
                        Ok(AnyParser::Type(Self::Input(Some(result_type))))
                    },
                ))
            }

            TypeParser::Input(_) => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::LEBU32(LEBParser::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::LEBU32(leb) = last_state else {
                        unreachable!();
                    };

                    let AnyParser::Type(Self::Input(result_type)) = this_state else {
                        unreachable!();
                    };
                    let entry_count = leb.production(irgen)?;

                    Ok(AnyParser::Type(if entry_count == 0 {
                        Self::Output(result_type, None)
                    } else {
                        Self::OutputSize(result_type, entry_count)
                    }))
                },
            )),
            TypeParser::OutputSize(_, size) => {
                let size = *size as usize;

                Ok(Advancement::YieldTo(
                    window.offset(),
                    AnyParser::Accumulate(Accumulator::new(size)),
                    |irgen, last_state, this_state| {
                        let AnyParser::Accumulate(accum) = last_state else {
                            unreachable!()
                        };
                        let AnyParser::Type(Self::OutputSize(input_result_type, _)) = this_state
                        else {
                            unreachable!();
                        };

                        let output_buf = accum.production(irgen)?;
                        let result_type = Self::map_buffer_to_result_type(&output_buf)?;

                        Ok(AnyParser::Type(Self::Output(
                            input_result_type,
                            Some(result_type),
                        )))
                    },
                ))
            }
            TypeParser::Output(_, _) => Ok(Advancement::Ready(window.offset())),
        }
    }

    fn production(self, _irgen: &mut T) -> Result<Self::Production, ParseError> {
        let Self::Output(_params, _returns) = self else {
            unreachable!();
        };

        todo!()
        /* Ok(Type(
            params.unwrap_or_default(),
            returns.unwrap_or_default(),
        ))*/
    }
}
