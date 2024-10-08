use uuasm_ir::IR;

use crate::{
    window::{AdvancementError, DecodeWindow},
    Advancement, IRError, Parse, ParseErrorKind, ParseResult,
};

use super::{accumulator::Accumulator, any::AnyParser};

#[derive(Default)]
pub enum ModuleParser<T: IR> {
    #[default]
    Magic,
    TakeSection(Vec<<T as IR>::Section>),
}

impl<T: IR> Parse<T> for ModuleParser<T> {
    type Production = <T as IR>::Module;

    fn advance(&mut self, _irgen: &mut T, window: &mut DecodeWindow) -> ParseResult<T> {
        match self {
            ModuleParser::Magic => Ok(Advancement::YieldTo(
                AnyParser::Accumulate(Accumulator::new(8)),
                |irgen, last_state, this_state| {
                    let AnyParser::Accumulate(accum) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        }
                    };
                    let AnyParser::Module(_) = this_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        }
                    };
                    let production = accum.production(irgen)?;

                    let magic = &production[0..4];
                    if magic != b"\x00asm" {
                        return Err(ParseErrorKind::BadMagic(u32::from_ne_bytes(
                            magic.try_into().unwrap(),
                        )));
                    }

                    let version = &production[4..];
                    if version != b"\x01\x00\x00\x00" {
                        return Err(ParseErrorKind::UnexpectedVersion(u32::from_le_bytes(
                            magic.try_into().unwrap(),
                        )));
                    }

                    Ok(AnyParser::Module(ModuleParser::TakeSection(Vec::new())))
                },
            )),

            ModuleParser::TakeSection(_) => {
                match window.peek() {
                    Err(AdvancementError::Expected(1, _)) => {
                        return Ok(Advancement::Ready);
                    }
                    Err(err) => return Err(err.into()),
                    _ => {}
                }

                Ok(Advancement::YieldTo(
                    AnyParser::Section(Default::default()),
                    |irgen, last_state, this_state| {
                        let AnyParser::Section(section) = last_state else {
                            unsafe {
                                crate::cold();
                                std::hint::unreachable_unchecked()
                            };
                        };
                        let AnyParser::Module(ModuleParser::TakeSection(mut sections)) = this_state
                        else {
                            unsafe {
                                crate::cold();
                                std::hint::unreachable_unchecked()
                            };
                        };

                        sections.push(section.production(irgen)?);

                        Ok(AnyParser::Module(ModuleParser::TakeSection(sections)))
                    },
                ))
            }
        }
    }

    fn production(self, irgen: &mut T) -> Result<Self::Production, ParseErrorKind<T::Error>> {
        let ModuleParser::TakeSection(sections) = self else {
            unsafe {
                crate::cold();
                std::hint::unreachable_unchecked()
            };
        };

        Ok(irgen.make_module(sections).map_err(IRError)?)
    }
}
