use uuasm_nodes::{Import, ImportDesc, Name};

use crate::{
    parser::{any::AnyParser, names::NameParser},
    Advancement, Parse, ParseError,
};

use super::importdescs::ImportDescParser;

#[derive(Default)]
pub enum ImportParser {
    #[default]
    Init,
    GotModule(Name),
    GotModuleAndName(Name, Name),
    Ready(Name, Name, ImportDesc),
}

impl Parse for ImportParser {
    type Production = Import;

    fn advance(&mut self, window: crate::window::DecodeWindow) -> crate::ParseResult {
        match self {
            ImportParser::Init => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::Name(NameParser::default()),
                |last_state, _| {
                    let AnyParser::Name(name) = last_state else {
                        unreachable!();
                    };

                    let name = name.production()?;
                    Ok(AnyParser::Import(Self::GotModule(name)))
                },
            )),

            ImportParser::GotModule(_) => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::Name(NameParser::default()),
                |last_state, this_state| {
                    let AnyParser::Name(name) = last_state else {
                        unreachable!();
                    };

                    let AnyParser::Import(ImportParser::GotModule(modname)) = this_state else {
                        unreachable!();
                    };
                    let name = name.production()?;
                    Ok(AnyParser::Import(Self::GotModuleAndName(modname, name)))
                },
            )),

            ImportParser::GotModuleAndName(_, _) => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::ImportDesc(ImportDescParser::default()),
                |last_state, this_state| {
                    let AnyParser::ImportDesc(desc) = last_state else {
                        unreachable!();
                    };

                    let AnyParser::Import(ImportParser::GotModuleAndName(modname, name)) =
                        this_state
                    else {
                        unreachable!();
                    };
                    let desc = desc.production()?;
                    Ok(AnyParser::Import(Self::Ready(modname, name, desc)))
                },
            )),

            ImportParser::Ready(_, _, _) => Ok(Advancement::Ready(window.offset())),
        }
    }

    fn production(self) -> Result<Self::Production, crate::ParseError> {
        let Self::Ready(modname, name, desc) = self else {
            return Err(ParseError::InvalidState(
                "Expected import to be in Ready state",
            ));
        };

        Ok(Import::new(modname, name, desc))
    }
}
