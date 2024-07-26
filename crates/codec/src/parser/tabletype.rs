use uuasm_nodes::IR;

use crate::{IRError, Parse};

use super::any::AnyParser;

#[derive(Default)]
pub enum TableTypeParser<T: IR> {
    #[default]
    Init,
    RefType(u8),
    Ready(u8, T::Limits),
}

impl<T: IR> Parse<T> for TableTypeParser<T> {
    type Production = T::TableType;

    fn advance(
        &mut self,
        _irgen: &mut T,
        mut window: crate::window::DecodeWindow,
    ) -> crate::ParseResult<T> {
        loop {
            *self = match self {
                TableTypeParser::Init => {
                    let candidate = window.take()?;
                    // TODO: should we validate the reftype here or should the ir be responsible
                    // for it? The argument for doing it here is that we're dealing with all of the
                    // parsing concerns here – not the reprs - but doing it in the generator lets
                    // us configure support for extra proposals.

                    Self::RefType(candidate)
                }
                TableTypeParser::RefType(_) => {
                    return Ok(crate::Advancement::YieldTo(
                        window.offset(),
                        AnyParser::Limits(Default::default()),
                        |irgen, last_state, this_state| {
                            let AnyParser::Limits(parser) = last_state else {
                                 unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                            };
                            let AnyParser::TableType(Self::RefType(rt)) = this_state else {
                                 unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                            };

                            let lims = parser.production(irgen)?;
                            Ok(AnyParser::TableType(Self::Ready(rt, lims)))
                        },
                    ))
                }
                TableTypeParser::Ready(_, _) => {
                    return Ok(crate::Advancement::Ready(window.offset()))
                }
            }
        }
    }

    fn production(
        self,
        irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseError<<T as IR>::Error>> {
        let Self::Ready(ref_type, limits) = self else {
             unsafe { crate::cold(); std::hint::unreachable_unchecked() }
        };

        Ok(irgen.make_table_type(ref_type, limits).map_err(IRError)?)
    }
}
