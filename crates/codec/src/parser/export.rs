use uuasm_nodes::IR;

use crate::{Advancement, IRError, Parse};

use super::any::AnyParser;

#[derive(Default)]
pub enum ExportParser<T: IR> {
    #[default]
    Init,

    Name(T::Name),
    Ready(T::Name, T::ExportDesc),
}

impl<T: IR> Parse<T> for ExportParser<T> {
    type Production = T::Export;

    fn advance(
        &mut self,
        _irgen: &mut T,
        window: crate::window::DecodeWindow,
    ) -> crate::ParseResult<T> {
        match self {
            Self::Init => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::Name(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::Name(parser) = last_state else {
                        unreachable!();
                    };

                    let name = parser.production(irgen)?;
                    Ok(AnyParser::Export(Self::Name(name)))
                },
            )),

            Self::Name(_) => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::ExportDesc(Default::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::ExportDesc(parser) = last_state else {
                        unreachable!();
                    };

                    let AnyParser::Export(Self::Name(name)) = this_state else {
                        unreachable!();
                    };

                    let desc = parser.production(irgen)?;
                    Ok(AnyParser::Export(Self::Ready(name, desc)))
                },
            )),

            Self::Ready(_, _) => Ok(Advancement::Ready(window.offset())),
        }
    }

    fn production(
        self,
        irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseError<<T as IR>::Error>> {
        let Self::Ready(name, desc) = self else {
            unreachable!()
        };

        Ok(irgen.make_export(name, desc).map_err(IRError)?)
    }
}
