use uuasm_nodes::IR;

use crate::{window::DecodeWindow, Advancement, IRError, Parse};

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

    fn advance(&mut self, _irgen: &mut T, window: &mut DecodeWindow) -> crate::ParseResult<T> {
        match self {
            Self::Init => Ok(Advancement::YieldTo(
                AnyParser::Name(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::Name(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let name = parser.production(irgen)?;
                    Ok(AnyParser::Export(Self::Name(name)))
                },
            )),

            Self::Name(_) => Ok(Advancement::YieldTo(
                AnyParser::ExportDesc(Default::default()),
                |irgen, last_state, this_state| {
                    let AnyParser::ExportDesc(parser) = last_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let AnyParser::Export(Self::Name(name)) = this_state else {
                        unsafe {
                            crate::cold();
                            std::hint::unreachable_unchecked()
                        };
                    };

                    let desc = parser.production(irgen)?;
                    Ok(AnyParser::Export(Self::Ready(name, desc)))
                },
            )),

            Self::Ready(_, _) => Ok(Advancement::Ready),
        }
    }

    fn production(
        self,
        irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseErrorKind<<T as IR>::Error>> {
        let Self::Ready(name, desc) = self else {
            unsafe {
                crate::cold();
                std::hint::unreachable_unchecked()
            }
        };

        Ok(irgen.make_export(name, desc).map_err(IRError)?)
    }
}
