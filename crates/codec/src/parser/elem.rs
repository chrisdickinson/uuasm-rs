use uuasm_nodes::IR;

use crate::{Advancement, IRError, Parse};

use super::any::AnyParser;

// ┌─── element type+exprs vs element kind + element idx
// │┌── explicit table index (or distinguishes passive from declarative)
// ││┌─ Passive or Declarative
// ↓↓↓
// 000: expr vec<funcidx>                      -> active
// 001: elemkind vec<funcidx>                  -> passive
// 010: tableidx expr elemkind vec<funcidx>    -> active
// 011: elemkind vec<funcidx>                  -> declarative
// 100: expr vec<expr>                         -> active
// 101: reftype vec<expr>                      -> passive
// 110: tableidx expr reftype vec<expr>        -> active
// 111: reftype vec<expr>                      -> declarative

// 000: expr vec<funcidx>                      -> active
// 010: tableidx expr elemkind vec<funcidx>    -> active
// 100: expr vec<expr>                         -> active
// 110: tableidx expr reftype vec<expr>        -> active
// 001: elemkind vec<funcidx>                  -> passive
// 011: elemkind vec<funcidx>                  -> declarative
// 101: reftype vec<expr>                      -> passive
// 111: reftype vec<expr>                      -> declarative
#[derive(Default)]
pub enum ElemParser<T: IR> {
    #[default]
    Init,

    Ready(T::Elem),
}

impl<T: IR> Parse<T> for ElemParser<T> {
    type Production = T::Elem;

    fn advance(
        &mut self,
        _irgen: &mut T,
        _window: crate::window::DecodeWindow,
    ) -> crate::ParseResult<T> {
        todo!()
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
