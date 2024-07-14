use uuasm_nodes::IR;

use crate::{
    window::{AdvancementError, DecodeWindow},
    Advancement, IRError, Parse, ParseError, ParseResult,
};

use super::{accumulator::Accumulator, any::AnyParser};

#[derive(Default)]
pub enum ModuleParser<T: IR> {
    #[default]
    Magic,
    TakeSection(Vec<<T as IR>::Section>),
    Done(Vec<<T as IR>::Section>),
}

impl<T: IR> Parse<T> for ModuleParser<T> {
    type Production = <T as IR>::Module;

    fn advance(&mut self, _irgen: &mut T, window: DecodeWindow) -> ParseResult<T> {
        match self {
            ModuleParser::Magic => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::Accumulate(Accumulator::new(8)),
                |irgen, last_state, this_state| {
                    let AnyParser::Accumulate(accum) = last_state else {
                        unreachable!()
                    };
                    let AnyParser::Module(_) = this_state else {
                        unreachable!()
                    };
                    let production = accum.production(irgen)?;

                    let magic = &production[0..4];
                    if magic != b"\x00asm" {
                        return Err(ParseError::BadMagic(u32::from_ne_bytes(
                            magic.try_into().unwrap(),
                        )));
                    }

                    let version = &production[4..];
                    if version != b"\x01\x00\x00\x00" {
                        return Err(ParseError::UnexpectedVersion(u32::from_le_bytes(
                            magic.try_into().unwrap(),
                        )));
                    }

                    Ok(AnyParser::Module(ModuleParser::TakeSection(Vec::new())))
                },
            )),

            ModuleParser::TakeSection(builder) => {
                match window.peek() {
                    Err(AdvancementError::Expected(1, _)) => {
                        *self = ModuleParser::Done(builder.split_off(0));
                        return Ok(Advancement::Ready(window.offset()));
                    }
                    Err(err) => return Err(err.into()),
                    _ => {}
                }

                Ok(Advancement::YieldTo(
                    window.offset(),
                    AnyParser::Section(Default::default()),
                    |irgen, last_state, this_state| {
                        let AnyParser::Section(section) = last_state else {
                            unreachable!();
                        };
                        let AnyParser::Module(ModuleParser::TakeSection(mut sections)) = this_state
                        else {
                            unreachable!();
                        };

                        sections.push(section.production(irgen)?);

                        Ok(AnyParser::Module(ModuleParser::TakeSection(sections)))
                    },
                ))
            }

            ModuleParser::Done(_) => Ok(Advancement::Ready(window.offset())),
        }
    }

    fn production(self, irgen: &mut T) -> Result<Self::Production, ParseError<T::Error>> {
        let ModuleParser::Done(sections) = self else {
            unreachable!();
        };

        Ok(irgen.make_module(sections).map_err(IRError)?)
    }
}
