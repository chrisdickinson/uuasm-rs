use uuasm_nodes::IR;

use crate::{window::DecodeWindow, Advancement, AnyParser, Parse, ParseErrorKind};

type Leader = u8;
type Ident = u32;
type Arg = u32;

pub enum InstrArgMultibyteParser {
    Init(Leader),
    Ident(Leader, Ident),
    IdentArity1(Leader, Ident, Arg),
    IdentArity2(Leader, Ident, Arg, Arg),
    IdentArity3(Leader, Ident, Arg, Arg, Arg),
    IdentArity4(Leader, Ident, Arg, Arg, Arg, Arg),
}

impl std::convert::From<u8> for InstrArgMultibyteParser {
    fn from(value: u8) -> Self {
        Self::Init(value)
    }
}

impl<T: IR> Parse<T> for InstrArgMultibyteParser {
    type Production = Self;

    fn advance(&mut self, _irgen: &mut T, window: &mut DecodeWindow) -> crate::ParseResult<T> {
        Ok(match self {
            Self::Init(leader) => Advancement::YieldTo(
                AnyParser::LEBU32(Default::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::LEBU32(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let AnyParser::ArgMultibyte(Self::Init(leader)) = this_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let ident = parser.production(irgen)?;
                    Ok(AnyParser::ArgMultibyte(Self::Ident(leader, ident)))
                },
            ),

            // conversions: no args
            Self::Ident(0xfc, 0x00..=0x07) => Advancement::Ready,

            Self::Ident(0xfc, 0x09 | 0x0b | 0x0d | 0x0f..=0x11) => Advancement::YieldTo(
                AnyParser::LEBU32(Default::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::LEBU32(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let AnyParser::ArgMultibyte(Self::Ident(leader, ident)) = this_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let arg0 = parser.production(irgen)?;
                    Ok(AnyParser::ArgMultibyte(Self::IdentArity1(
                        leader, ident, arg0,
                    )))
                },
            ),

            Self::IdentArity1(0xfc, 0x09 | 0x0b | 0x0d | 0x0f..=0x11, _) => Advancement::Ready,

            Self::Ident(0xfc, 0x08 | 0x0a..=0x0c | 0x0e) => Advancement::YieldTo(
                AnyParser::LEBU32(Default::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::LEBU32(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let AnyParser::ArgMultibyte(Self::Ident(leader, ident)) = this_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let arg0 = parser.production(irgen)?;
                    Ok(AnyParser::ArgMultibyte(Self::IdentArity1(
                        leader, ident, arg0,
                    )))
                },
            ),
            Self::IdentArity1(0xfc, 0x08 | 0x0a..=0x0c | 0x0e, _) => Advancement::YieldTo(
                AnyParser::LEBU32(Default::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::LEBU32(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let AnyParser::ArgMultibyte(Self::IdentArity1(leader, ident, arg0)) =
                        this_state
                    else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let arg1 = parser.production(irgen)?;
                    Ok(AnyParser::ArgMultibyte(Self::IdentArity2(
                        leader, ident, arg0, arg1,
                    )))
                },
            ),
            Self::IdentArity2(0xfc, 0x08 | 0x0a..=0x0c | 0x0e, _, _) => Advancement::Ready,

            // 1 arg
            Self::Ident(0xfd, 0x00..=0x0b | 92 | 93) => Advancement::YieldTo(
                AnyParser::LEBU32(Default::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::LEBU32(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let AnyParser::ArgMultibyte(Self::IdentArity1(leader, ident, arg0)) =
                        this_state
                    else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let arg1 = parser.production(irgen)?;
                    Ok(AnyParser::ArgMultibyte(Self::IdentArity2(
                        leader, ident, arg0, arg1,
                    )))
                },
            ),
            Self::IdentArity1(0xfd, 0x00..=0x0b | 92 | 93, _) => Advancement::Ready,

            // 2 args (1 memarg, 1 lane)
            Self::Ident(0xfd, 0x15..=0x22 | 0x54..=0x5b) => Advancement::YieldTo(
                AnyParser::LEBU32(Default::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::LEBU32(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let AnyParser::ArgMultibyte(Self::IdentArity1(leader, ident, arg0)) =
                        this_state
                    else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let arg1 = parser.production(irgen)?;
                    Ok(AnyParser::ArgMultibyte(Self::IdentArity2(
                        leader, ident, arg0, arg1,
                    )))
                },
            ),
            Self::IdentArity1(0xfd, instr @ (0x15..=0x22 | 0x54..=0x5b), arg0) => {
                *self = Self::IdentArity2(0xfd, *instr, *arg0, window.take()? as u32);
                Advancement::Ready
            }

            // 0 args
            Self::Ident(0xfd, 0x0e..=0x14 | 0x23..0xff) => Advancement::Ready,

            _ => todo!(),
        })
    }

    fn production(
        self,
        _irgen: &mut T,
    ) -> Result<Self::Production, ParseErrorKind<<T as IR>::Error>> {
        Ok(self)
    }
}
