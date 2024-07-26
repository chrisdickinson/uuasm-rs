use uuasm_nodes::IR;

use crate::{Advancement, IRError, Parse};

use super::any::AnyParser;

#[derive(Default)]
pub enum LocalParser<T: IR> {
    #[default]
    Init,

    Count(u32),

    Ready(T::Local),
}

impl<T: IR> Parse<T> for LocalParser<T> {
    type Production = T::Local;

    fn advance(
        &mut self,
        irgen: &mut T,
        mut window: crate::window::DecodeWindow,
    ) -> crate::ParseResult<T> {
        match self {
            LocalParser::Init => Ok(Advancement::YieldTo(
                window.offset(),
                AnyParser::LEBU32(Default::default()),
                |irgen, last_state, _| {
                    let AnyParser::LEBU32(parser) = last_state else {
                         unsafe { crate::cold(); std::hint::unreachable_unchecked() }
                    };

                    Ok(AnyParser::Local(Self::Count(parser.production(irgen)?)))
                },
            )),
            LocalParser::Count(count) => {
                let candidate = window.take()?;
                let val_type = irgen.make_val_type(candidate).map_err(IRError)?;
                let local = irgen.make_local(*count, val_type).map_err(IRError)?;
                *self = LocalParser::Ready(local);
                Ok(Advancement::Ready(window.offset()))
            }
            LocalParser::Ready(_) => Ok(Advancement::Ready(window.offset())),
        }
    }

    fn production(
        self,
        _irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseError<<T as IR>::Error>> {
        let Self::Ready(production) = self else {
             unsafe { crate::cold(); std::hint::unreachable_unchecked() }
        };

        Ok(production)
    }
}
