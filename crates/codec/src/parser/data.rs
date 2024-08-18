use uuasm_ir::IR;

use crate::{window::DecodeWindow, Advancement, IRError, Parse, ParseErrorKind};

use super::any::AnyParser;

#[derive(Default)]
pub enum DataParser<T: IR> {
    #[default]
    Init,
    MemIdx(T::MemIdx),
    Expr(T::MemIdx, T::Expr),

    Ready(T::Data),
}

impl<T: IR> Parse<T> for DataParser<T> {
    type Production = T::Data;

    fn advance(&mut self, _irgen: &mut T, window: &mut DecodeWindow) -> crate::ParseResult<T> {
        match self {
            Self::Init => match window.take()? {
                0x00 => Ok(Advancement::YieldTo(
                    AnyParser::Expr(Default::default()),
                    // this parses a constexpr, then a bytevec, becoming a data::active(vec, memidx(0),
                    // expr)
                    |irgen, last_state, _| {
                        let AnyParser::Expr(parser) = last_state else {
                            unsafe {
                                crate::cold();
                                std::hint::unreachable_unchecked()
                            }
                        };

                        let expr = parser.production(irgen)?;
                        let mem_idx = irgen.make_mem_index(0).map_err(IRError)?;

                        Ok(AnyParser::Data(Self::Expr(mem_idx, expr)))
                    },
                )),

                // this parses a bytevec and becomes a data::passive
                0x01 => Ok(Advancement::YieldTo(
                    AnyParser::ByteVec(Default::default()),
                    // this parses a constexpr, then a bytevec, becoming a data::active(vec, memidx(0),
                    // expr)
                    |irgen, last_state, _| {
                        let AnyParser::ByteVec(parser) = last_state else {
                            unsafe {
                                crate::cold();
                                std::hint::unreachable_unchecked()
                            }
                        };

                        let data = parser.production(irgen)?;
                        let data = irgen.make_data_passive(data).map_err(IRError)?;

                        Ok(AnyParser::Data(Self::Ready(data)))
                    },
                )),

                // this parses a memory index, const expr, and bytevec; producing
                // an active data segment
                0x02 => Ok(Advancement::YieldTo(
                    AnyParser::LEBU32(Default::default()),
                    // this parses a constexpr, then a bytevec, becoming a data::active(vec, memidx(0),
                    // expr)
                    |irgen, last_state, _| {
                        let AnyParser::LEBU32(parser) = last_state else {
                            unsafe {
                                crate::cold();
                                std::hint::unreachable_unchecked()
                            }
                        };

                        let mem_idx = parser.production(irgen)?;
                        let mem_idx = irgen.make_mem_index(mem_idx).map_err(IRError)?;

                        Ok(AnyParser::Data(Self::MemIdx(mem_idx)))
                    },
                )),

                unk => Err(ParseErrorKind::BadDataType(unk)),
            },

            Self::MemIdx(_) => Ok(Advancement::YieldTo(
                AnyParser::Expr(Default::default()),
                // this parses a constexpr, then a bytevec, becoming a data::active(vec, memidx(0),
                // expr)
                |irgen, last_state, this_state| {
                    let AnyParser::Expr(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        }
                    };

                    let AnyParser::Data(Self::MemIdx(mem_idx)) = this_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        }
                    };

                    let expr = parser.production(irgen)?;

                    Ok(AnyParser::Data(Self::Expr(mem_idx, expr)))
                },
            )),

            Self::Expr(_, _) => {
                Ok(Advancement::YieldTo(
                    AnyParser::ByteVec(Default::default()),
                    // this parses a constexpr, then a bytevec, becoming a data::active(vec, memidx(0),
                    // expr)
                    |irgen, last_state, this_state| {
                        let AnyParser::ByteVec(parser) = last_state else {
                            unsafe {
                                crate::cold();
                                std::hint::unreachable_unchecked()
                            }
                        };
                        let AnyParser::Data(Self::Expr(mem_idx, expr)) = this_state else {
                            unsafe {
                                crate::cold();
                                std::hint::unreachable_unchecked()
                            }
                        };

                        let data = parser.production(irgen)?;
                        let data = irgen
                            .make_data_active(data, mem_idx, expr)
                            .map_err(IRError)?;

                        Ok(AnyParser::Data(Self::Ready(data)))
                    },
                ))
            }
            Self::Ready(_) => Ok(Advancement::Ready),
        }
    }

    fn production(
        self,
        _irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseErrorKind<<T as IR>::Error>> {
        let Self::Ready(production) = self else {
            unsafe {
                crate::cold();
                std::hint::unreachable_unchecked()
            }
        };

        Ok(production)
    }
}
