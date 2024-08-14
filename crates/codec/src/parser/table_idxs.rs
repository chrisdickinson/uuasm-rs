use uuasm_nodes::IR;

use crate::{window::DecodeWindow, Advancement, IRError, Parse};

use super::any::AnyParser;

pub struct TableIdxParser<T: IR>(Option<T::TableIdx>);

impl<T: IR> Default for TableIdxParser<T> {
    fn default() -> Self {
        Self(None)
    }
}

impl<T: IR> Parse<T> for TableIdxParser<T> {
    type Production = T::TableIdx;

    fn advance(&mut self, _irgen: &mut T, window: &mut DecodeWindow) -> crate::ParseResult<T> {
        if self.0.is_some() {
            return Ok(Advancement::Ready);
        }
        Ok(Advancement::YieldTo(
            AnyParser::LEBU32(Default::default()),
            |irgen, last_state, _| {
                let AnyParser::LEBU32(parser) = last_state else {
                    unsafe {
                        crate::cold();
                        std::hint::unreachable_unchecked()
                    };
                };

                let idx = parser.production(irgen)?;
                let idx = irgen.make_table_index(idx).map_err(IRError)?;

                Ok(AnyParser::TableIdx(Self(Some(idx))))
            },
        ))
    }

    fn production(
        self,
        _irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseErrorKind<<T as IR>::Error>> {
        let Self(Some(production)) = self else {
            unsafe {
                crate::cold();
                std::hint::unreachable_unchecked()
            }
        };

        Ok(production)
    }
}
