use uuasm_nodes::IR;

use crate::{Advancement, IRError, Parse};

use super::any::AnyParser;

pub struct FuncIdxParser<T: IR>(Option<T::FuncIdx>);

impl<T: IR> Default for FuncIdxParser<T> {
    fn default() -> Self {
        Self(None)
    }
}

impl<T: IR> Parse<T> for FuncIdxParser<T> {
    type Production = T::FuncIdx;

    fn advance(
        &mut self,
        _irgen: &mut T,
        window: crate::window::DecodeWindow,
    ) -> crate::ParseResult<T> {
        if self.0.is_some() {
            return Ok(Advancement::Ready(window.offset()));
        }
        Ok(Advancement::YieldTo(
            window.offset(),
            AnyParser::LEBU32(Default::default()),
            |irgen, last_state, _| {
                let AnyParser::LEBU32(parser) = last_state else {
                     unsafe { crate::cold(); std::hint::unreachable_unchecked() };
                };

                let idx = parser.production(irgen)?;
                let idx = irgen.make_func_index(idx).map_err(IRError)?;

                Ok(AnyParser::FuncIdx(Self(Some(idx))))
            },
        ))
    }

    fn production(
        self,
        _irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseError<<T as IR>::Error>> {
        let Self(Some(production)) = self else {
             unsafe { crate::cold(); std::hint::unreachable_unchecked() }
        };

        Ok(production)
    }
}
