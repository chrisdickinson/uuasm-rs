use uuasm_ir::IR;

use crate::{window::DecodeWindow, Advancement, IRError, Parse};

use super::any::AnyParser;

pub struct MemIdxParser<T: IR>(Option<T::MemIdx>);

impl<T: IR> Default for MemIdxParser<T> {
    fn default() -> Self {
        Self(None)
    }
}

impl<T: IR> Parse<T> for MemIdxParser<T> {
    type Production = T::MemIdx;

    fn advance(&mut self, _irgen: &mut T, _window: &mut DecodeWindow) -> crate::ParseResult<T> {
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
                let idx = irgen.make_mem_index(idx).map_err(IRError)?;

                Ok(AnyParser::MemIdx(Self(Some(idx))))
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
