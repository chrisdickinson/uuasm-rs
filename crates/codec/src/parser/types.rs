use uuasm_ir::IR;

use crate::{window::DecodeWindow, Advancement, IRError, Parse, ParseErrorKind, ParseResult};

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

impl<T: IR> Parse<T> for TypeParser<T> {
    type Production = <T as IR>::Type;

    fn advance(&mut self, _irgen: &mut T, window: &mut DecodeWindow) -> ParseResult<T> {
        match self {
            TypeParser::Init => {
                let tag = window.take()?;
                if tag != 0x60 {
                    return Err(ParseErrorKind::BadTypePrefix(tag));
                }

                Ok(Advancement::YieldTo(
                    AnyParser::LEBU32(LEBParser::default()),
                    |irgen, last_state, this_state| {
                        let AnyParser::LEBU32(leb) = last_state else {
                            unsafe {
                                crate::cold();
                                std::hint::unreachable_unchecked()
                            };
                        };

                        let AnyParser::Type(Self::Init) = this_state else {
                            unsafe {
                                crate::cold();
                                std::hint::unreachable_unchecked()
                            };
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
                    AnyParser::Accumulate(Accumulator::new(size)),
                    |irgen, last_state, this_state| {
                        let AnyParser::Accumulate(accum) = last_state else {
                            unsafe {
                                crate::cold();
                                std::hint::unreachable_unchecked()
                            }
                        };
                        let AnyParser::Type(Self::InputSize(_)) = this_state else {
                            unsafe {
                                crate::cold();
                                std::hint::unreachable_unchecked()
                            };
                        };

                        let input_buf = accum.production(irgen)?;
                        let result_type = irgen.make_result_type(&input_buf).map_err(IRError)?;
                        Ok(AnyParser::Type(Self::Input(Some(result_type))))
                    },
                ))
            }

            TypeParser::Input(_) => Ok(Advancement::YieldTo(
                AnyParser::LEBU32(LEBParser::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::LEBU32(leb) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let AnyParser::Type(Self::Input(result_type)) = this_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
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
                    AnyParser::Accumulate(Accumulator::new(size)),
                    |irgen, last_state, this_state| {
                        let AnyParser::Accumulate(accum) = last_state else {
                            unsafe {
                                crate::cold();
                                std::hint::unreachable_unchecked()
                            }
                        };
                        let AnyParser::Type(Self::OutputSize(input_result_type, _)) = this_state
                        else {
                            unsafe {
                                crate::cold();
                                std::hint::unreachable_unchecked()
                            };
                        };

                        let output_buf = accum.production(irgen)?;
                        let result_type = irgen.make_result_type(&output_buf).map_err(IRError)?;

                        Ok(AnyParser::Type(Self::Output(
                            input_result_type,
                            Some(result_type),
                        )))
                    },
                ))
            }
            TypeParser::Output(_, _) => Ok(Advancement::Ready),
        }
    }

    fn production(self, irgen: &mut T) -> Result<Self::Production, ParseErrorKind<T::Error>> {
        let Self::Output(params, returns) = self else {
            unsafe {
                crate::cold();
                std::hint::unreachable_unchecked()
            };
        };

        Ok(irgen.make_func_type(params, returns).map_err(IRError)?)
    }
}
