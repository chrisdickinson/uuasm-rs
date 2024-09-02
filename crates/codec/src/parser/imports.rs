use uuasm_ir::IR;

use crate::{
    parser::{any::AnyParser, names::NameParser},
    window::DecodeWindow,
    Advancement, IRError, Parse, ParseErrorKind,
};

use super::importdescs::ImportDescParser;

#[derive(Default)]
pub enum ImportParser<T: IR> {
    #[default]
    Init,
    GotModule(<T as IR>::Name),
    GotModuleAndName(<T as IR>::Name, <T as IR>::Name),
    Ready(<T as IR>::Name, <T as IR>::Name, <T as IR>::ImportDesc),
}

impl<T: IR> Parse<T> for ImportParser<T> {
    type Production = <T as IR>::Import;

    fn advance(&mut self, _irgen: &mut T, _window: &mut DecodeWindow) -> crate::ParseResult<T> {
        match self {
            ImportParser::Init => Ok(Advancement::YieldTo(
                AnyParser::Name(NameParser::default()),
                |irgen, last_state, _| {
                    let AnyParser::Name(name) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let name = name.production(irgen)?;
                    Ok(AnyParser::Import(Self::GotModule(name)))
                },
            )),

            ImportParser::GotModule(_) => Ok(Advancement::YieldTo(
                AnyParser::Name(NameParser::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::Name(name) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let AnyParser::Import(ImportParser::GotModule(modname)) = this_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let name = name.production(irgen)?;
                    Ok(AnyParser::Import(Self::GotModuleAndName(modname, name)))
                },
            )),

            ImportParser::GotModuleAndName(_, _) => Ok(Advancement::YieldTo(
                AnyParser::ImportDesc(ImportDescParser::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::ImportDesc(desc) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let AnyParser::Import(ImportParser::GotModuleAndName(modname, name)) =
                        this_state
                    else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };
                    let desc = desc.production(irgen)?;
                    Ok(AnyParser::Import(Self::Ready(modname, name, desc)))
                },
            )),

            ImportParser::Ready(_, _, _) => Ok(Advancement::Ready),
        }
    }

    fn production(
        self,
        irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseErrorKind<T::Error>> {
        let Self::Ready(modname, name, desc) = self else {
            return Err(ParseErrorKind::InvalidState(
                "Expected import to be in Ready state",
            ));
        };

        Ok(irgen.make_import(modname, name, desc).map_err(IRError)?)
    }
}
