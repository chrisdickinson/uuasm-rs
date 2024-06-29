use uuasm_nodes::{Module, ModuleBuilder, SectionType};

use crate::{window::DecodeWindow, Advancement, Parse, ParseError, ParseResult};

use super::{accumulator::Accumulator, state::ParseState, take::Take};

#[derive(Default)]
pub enum ModuleParser {
    #[default]
    Magic,
    TakeSection(Box<Option<ModuleBuilder>>),
    Done(Box<Module>),
}

impl Parse for ModuleParser {
    type Production = Module;

    fn advance(&mut self, window: DecodeWindow) -> ParseResult {
        match self {
            ModuleParser::Magic => Ok(Advancement::YieldTo(
                window.offset(),
                ParseState::Accumulate(Take::new(Accumulator::new(8), 8)),
                |last_state, this_state| {
                    let ParseState::Accumulate(accum) = last_state else {
                        unreachable!()
                    };
                    let ParseState::Module(_) = this_state else {
                        unreachable!()
                    };
                    let production = accum.production()?;

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

                    Ok(ParseState::Module(ModuleParser::TakeSection(Box::new(
                        Some(ModuleBuilder::new()),
                    ))))
                },
            )),

            ModuleParser::TakeSection(builder) => {
                match window.peek() {
                    Err(ParseError::Expected(1)) => {
                        let builder = builder.take().unwrap();
                        *self = ModuleParser::Done(Box::new(builder.build()));
                        return Ok(Advancement::Ready(window.offset()));
                    }
                    Err(err) => return Err(err),
                    _ => {}
                }

                Ok(Advancement::YieldTo(
                    window.offset(),
                    ParseState::Section(Default::default()),
                    |last_state, this_state| {
                        let ParseState::Section(section) = last_state else {
                            unreachable!();
                        };
                        let ParseState::Module(ModuleParser::TakeSection(mut builder_box)) =
                            this_state
                        else {
                            unreachable!();
                        };

                        let builder = builder_box.take().unwrap();
                        let section_type = section.production()?;
                        builder_box.replace(match section_type {
                            SectionType::Custom(xs) => builder.custom_section(xs),
                            SectionType::Type(xs) => builder.type_section(xs),
                            SectionType::Import(_) => todo!(),
                            SectionType::Function(_) => todo!(),
                            SectionType::Table(_) => todo!(),
                            SectionType::Memory(_) => todo!(),
                            SectionType::Global(_) => todo!(),
                            SectionType::Export(_) => todo!(),
                            SectionType::Start(_) => todo!(),
                            SectionType::Element(_) => todo!(),
                            SectionType::Code(_) => todo!(),
                            SectionType::Data(_) => todo!(),
                            SectionType::DataCount(_) => todo!(),
                        });

                        Ok(ParseState::Module(ModuleParser::TakeSection(builder_box)))
                    },
                ))
            }

            ModuleParser::Done(_) => Ok(Advancement::Ready(window.offset())),
        }
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        let ModuleParser::Done(module) = self else {
            unreachable!();
        };

        Ok(*module)
    }
}
