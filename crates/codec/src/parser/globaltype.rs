use uuasm_nodes::IR;

use crate::{IRError, Parse, ParseError};

#[derive(Default)]
pub enum GlobalTypeParser<T: IR> {
    #[default]
    Init,
    ValType(Option<T::ValType>),
    Ready(T::ValType, bool),
}

impl<T: IR> Parse<T> for GlobalTypeParser<T> {
    type Production = T::GlobalType;

    fn advance(
        &mut self,
        irgen: &mut T,
        mut window: crate::window::DecodeWindow,
    ) -> crate::ParseResult<T> {
        loop {
            *self = match self {
                GlobalTypeParser::Init => {
                    let candidate = window.take()?;
                    let val_type = irgen.make_val_type(candidate).map_err(IRError)?;
                    Self::ValType(Some(val_type))
                }
                GlobalTypeParser::ValType(val_type) => {
                    let mutability = match window.take()? {
                        0x00 => false,
                        0x01 => true,
                        unk => return Err(ParseError::BadMutability(unk)),
                    };

                    Self::Ready(val_type.take().unwrap(), mutability)
                }
                GlobalTypeParser::Ready(_, _) => {
                    return Ok(crate::Advancement::Ready(window.offset()))
                }
            }
        }
    }

    fn production(
        self,
        irgen: &mut T,
    ) -> Result<Self::Production, crate::ParseError<<T as IR>::Error>> {
        let Self::Ready(val_type, mutability) = self else {
            unreachable!()
        };

        Ok(irgen
            .make_global_type(val_type, mutability)
            .map_err(IRError)?)
    }
}
