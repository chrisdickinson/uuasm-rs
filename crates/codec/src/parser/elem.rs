use std::mem;

use uuasm_ir::IR;

use crate::{window::DecodeWindow, Advancement, IRError, Parse};

use super::any::AnyParser;

pub enum ElementMode<T: IR> {
    Passive,
    Active(T::TableIdx, T::Expr),
    Declarative,
}

// Elem is { reftype, init(vec expr), mode }
// mode is passive | declarative | active { tableidx, offset expr }

// ┌─── element type+exprs vs element kind + element idx; vec of func idx or vec of expr
// │┌── if low bit is 1: passive or declarative; if low bit is 0: whether or not we have a tableidx
// ││┌─ Passive or Declarative
// ↓↓↓
// 000: expr vec<funcidx>                      -> active
// 001: elemkind vec<funcidx>                  -> passive
// 010: tableidx expr elemkind vec<funcidx>    -> active
// 011: elemkind vec<funcidx>                  -> declarative
// 100: expr vec<expr>                         -> active
// 101: reftype vec<expr>                      -> passive
// 110: tableidx expr reftype vec<expr>        -> active
// 111: reftype vec<expr>                      -> declarative

// 000: expr vec<funcidx>                      -> active
// 010: tableidx expr elemkind vec<funcidx>    -> active
// 100: expr vec<expr>                         -> active
// 110: tableidx expr reftype vec<expr>        -> active
// 001: elemkind vec<funcidx>                  -> passive
// 011: elemkind vec<funcidx>                  -> declarative
// 101: reftype vec<expr>                      -> passive
// 111: reftype vec<expr>                      -> declarative

// 000: expr vec<funcidx>                      -> active
// 010: tableidx expr elemkind vec<funcidx>    -> active
// 100: expr vec<expr>                         -> active
// 110: tableidx expr reftype vec<expr>        -> active
// 001: elemkind vec<funcidx>                  -> passive
// 011: elemkind vec<funcidx>                  -> declarative
// 101: reftype vec<expr>                      -> passive
// 111: reftype vec<expr>                      -> declarative
#[derive(Default)]
pub enum ElemParser<T: IR> {
    #[default]
    Init,

    Flags(u8),

    ParseActiveModeTableIdx(u8, T::TableIdx),

    ParseMode(u8, ElementMode<T>),
    ParseElemKind(u8, T::ElemMode, Option<u32>),
    ParseRefType(u8, T::ElemMode, Option<T::RefType>),

    Ready(T::Elem),
}

impl<T: IR> Parse<T> for ElemParser<T> {
    type Production = T::Elem;

    fn advance(&mut self, irgen: &mut T, window: &mut DecodeWindow) -> crate::ParseResult<T> {
        'restart: loop {
            return Ok(match self {
                Self::Init => {
                    let flags = window.take()?;

                    *self = Self::Flags(flags);
                    if flags & 0b11 == 0b10 {
                        Advancement::YieldTo(
                            AnyParser::LEBU32(Default::default()),
                            |irgen, last_state, this_state| {
                                let AnyParser::LEBU32(parser) = last_state else {
                                    unreachable!();
                                };
                                let AnyParser::Elem(Self::Flags(flags)) = this_state else {
                                    unreachable!();
                                };

                                let candidate = parser.production(irgen)?;

                                Ok(AnyParser::Elem(Self::ParseActiveModeTableIdx(
                                    flags,
                                    irgen.make_table_index(candidate).map_err(IRError)?,
                                )))
                            },
                        )
                    } else if flags & 0b1 == 0 {
                        *self = Self::ParseActiveModeTableIdx(
                            flags,
                            irgen.make_table_index(0).map_err(IRError)?,
                        );
                        continue 'restart;
                    } else {
                        let mode = if flags & 0b010 == 0 {
                            ElementMode::Declarative
                        } else {
                            ElementMode::Passive
                        };

                        *self = Self::ParseMode(flags, mode);
                        continue 'restart;
                    }
                }

                Self::Flags(_) => unreachable!(),

                Self::ParseActiveModeTableIdx(_, _) => {
                    irgen.start_elem_active_table_index().map_err(IRError)?;

                    Advancement::YieldTo(
                        AnyParser::Expr(Default::default()),
                        |irgen, last_state, this_state| {
                            let AnyParser::Expr(parser) = last_state else {
                                unreachable!();
                            };
                            let AnyParser::Elem(Self::ParseActiveModeTableIdx(flags, table_idx)) =
                                this_state
                            else {
                                unreachable!();
                            };

                            let expr = parser.production(irgen)?;

                            // need to take one from the window lolol
                            Ok(AnyParser::Elem(Self::ParseMode(
                                flags,
                                ElementMode::Active(table_idx, expr),
                            )))
                        },
                    )
                }

                Self::ParseMode(_, _) => {
                    let Self::ParseMode(flags, mode) = mem::take(self) else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        }
                    };
                    let has_type = flags & 0b11 != 0;
                    if flags & 0b100 == 0 {
                        // elemkind vec<funcidx>
                        *self = Self::ParseElemKind(
                            flags,
                            make_elem_mode(irgen, mode)?,
                            if has_type {
                                Some(window.take()? as u32)
                            } else {
                                None
                            },
                        );
                        continue 'restart;
                    }
                    // reftype vec<expr>
                    let ref_type = if has_type {
                        let ref_type = irgen.make_ref_type(window.take()?).map_err(IRError)?;
                        Some(ref_type)
                    } else {
                        None
                    };
                    let mode = make_elem_mode(irgen, mode)?;
                    *self = Self::ParseRefType(flags, mode, ref_type);
                    continue 'restart;
                }

                Self::ParseElemKind(_, _, _) => Advancement::YieldTo(
                    AnyParser::RepeatedLEBU32(Default::default()),
                    |irgen, last_state, this_state| {
                        let AnyParser::RepeatedLEBU32(parser) = last_state else {
                            unreachable!();
                        };
                        let AnyParser::Elem(Self::ParseElemKind(flags, mode, kind)) = this_state
                        else {
                            unreachable!();
                        };

                        let func_indices = parser.production(irgen)?;
                        let elem = irgen
                            .make_elem_from_indices(kind, mode, func_indices, flags)
                            .map_err(IRError)?;
                        Ok(AnyParser::Elem(Self::Ready(elem)))
                    },
                ),

                Self::ParseRefType(_, _, ref_type) => {
                    irgen
                        .start_elem_reftype_list(ref_type.as_ref())
                        .map_err(IRError)?;

                    Advancement::YieldTo(
                        AnyParser::ExprList(Default::default()),
                        |irgen, last_state, this_state| {
                            let AnyParser::ExprList(parser) = last_state else {
                                unreachable!();
                            };
                            let AnyParser::Elem(Self::ParseRefType(flags, mode, kind)) = this_state
                            else {
                                unreachable!();
                            };

                            let expr_list = parser.production(irgen)?;
                            let elem = irgen
                                .make_elem_from_exprs(kind, mode, expr_list, flags)
                                .map_err(IRError)?;
                            Ok(AnyParser::Elem(Self::Ready(elem)))
                        },
                    )
                }

                Self::Ready(_) => Advancement::Ready,
            });
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

fn make_elem_mode<T: IR>(
    irgen: &mut T,
    mode: ElementMode<T>,
) -> Result<T::ElemMode, crate::ParseErrorKind<<T as IR>::Error>> {
    Ok(match mode {
        ElementMode::Passive => irgen.make_elem_mode_passive(),
        ElementMode::Active(table_idx, expr) => irgen.make_elem_mode_active(table_idx, expr),
        ElementMode::Declarative => irgen.make_elem_mode_declarative(),
    }
    .map_err(IRError)?)
}
