use std::marker::PhantomData;

use uuasm_nodes::{DefaultIRGenerator, IR};

use crate::{
    cold,
    parser::{
        any::{AnyParser, AnyProduction},
        module::ModuleParser,
    },
    window::{AdvancementError, DecodeWindow},
    Advancement, ExtractTarget, Parse, ParseError, ResumeFunc,
};

/*
 * A parser has a number of states, held in a stack. The last item in the stack is the current
 * parser. The parser can be written to, or flushed. When new bytes come in, the topmost stack
 * item processes them. If it is complete, it returns control to the next stack state along with
 * the production. The stack state can `take(N)`, `peek(N)`, `skip(N)`.
 */
pub struct Decoder<T: IR, Target: ExtractTarget<AnyProduction<T>>> {
    state: Vec<(AnyParser<T>, ResumeFunc<T>, Option<u32>)>,
    position: usize,
    irgen: T,
    _marker: PhantomData<Target>,
}

fn noop_resume<T: IR>(
    _irgen: &mut T,
    _last_state: AnyParser<T>,
    _this_state: AnyParser<T>,
) -> Result<AnyParser<T>, ParseError<T::Error>> {
    Err(ParseError::InvalidState("this state should be unreachable"))
}

impl Default for Decoder<DefaultIRGenerator, <DefaultIRGenerator as IR>::Module> {
    fn default() -> Self {
        Self::new(
            AnyParser::Module(ModuleParser::default()),
            DefaultIRGenerator::default(),
        )
    }
}

impl<T: IR, Target: ExtractTarget<AnyProduction<T>>> Decoder<T, Target> {
    pub fn new(parser: AnyParser<T>, irgen: T) -> Self {
        Self::new_with_position(parser, irgen, 0)
    }

    pub fn new_with_position(parser: AnyParser<T>, irgen: T, position: usize) -> Self {
        let mut state = Vec::with_capacity(if position > 0 { 4 } else { 16 });
        state.push((parser, noop_resume as ResumeFunc<T>, None));
        Self {
            state,
            position,
            irgen,
            _marker: PhantomData,
        }
    }

    fn write_inner<'a>(
        &'_ mut self,
        chunk: &'a [u8],
        eos: bool,
    ) -> Result<(Target, &'a [u8]), ParseError<T::Error>> {
        let offset = 0;
        let mut consumed = 0;
        loop {
            let (state, resume, bound) = self.state.last_mut().unwrap();
            let window = DecodeWindow::new(&chunk[consumed..], 0, self.position, eos, *bound);
            let offset = match state.advance(&mut self.irgen, window) {
                Ok(Advancement::Ready(offset)) => {
                    drop(state);
                    drop(resume);
                    drop(bound);
                    let (n_state, n_resume, n_bound) = self.state.pop().unwrap();
                    let Some((receiver, last_resume, last_bound)) = self.state.pop() else {
                        cold();
                        let output = Target::extract(n_state.production(&mut self.irgen)?)?;
                        return Ok((output, &chunk[offset..]));
                    };

                    let resumed = match n_resume(&mut self.irgen, n_state, receiver) {
                        Ok(v) => v,
                        Err(e) => {
                            self.state.push((
                                AnyParser::Failed(e.clone()),
                                noop_resume as ResumeFunc<T>,
                                None,
                            ));
                            return Err(e);
                        }
                    };

                    self.state.push((resumed, last_resume, last_bound));
                    offset
                }

                Ok(Advancement::YieldTo(new_offset, next_state, next_resume)) => {
                    if let Some(xs) = bound.as_mut() {
                        *xs = xs.saturating_sub((new_offset - offset) as u32);
                    }
                    drop(state);
                    drop(resume);
                    let bound = *bound;
                    self.state.push((next_state, next_resume, bound));
                    new_offset
                }

                Ok(Advancement::YieldToBounded(
                    new_offset,
                    next_bound,
                    next_state,
                    next_resume,
                )) => {
                    if let Some(xs) = bound.as_mut() {
                        *xs = xs.saturating_sub((new_offset - offset) as u32);
                        if next_bound > *xs {
                            let e = ParseError::UnexpectedEOS;
                            self.state.push((
                                AnyParser::Failed(e.clone()),
                                noop_resume as ResumeFunc<T>,
                                None,
                            ));
                            return Err(e);
                        }
                    }
                    self.state.push((next_state, next_resume, Some(next_bound)));
                    offset
                }

                Err(err @ ParseError::Advancement(AdvancementError::Incomplete(_))) => {
                    if let Some(xs) = bound.as_mut() {
                        *xs = xs.saturating_sub((chunk.len() - offset) as u32);
                    }
                    return Err(err);
                }

                Err(e) => {
                    self.state.push((
                        AnyParser::Failed(e.clone()),
                        noop_resume as ResumeFunc<T>,
                        None,
                    ));
                    return Err(e);
                }
            };
            consumed += offset;
            self.position += offset;
        }
    }

    pub fn write<'a>(
        &'_ mut self,
        chunk: &'a [u8],
    ) -> Result<(Target, &'a [u8]), ParseError<T::Error>> {
        self.write_inner(chunk, false)
    }

    pub fn flush(&mut self) -> Result<Target, ParseError<T::Error>> {
        let (output, _) = self.write_inner(&[], true)?;
        Ok(output)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parser2_works_basically() -> anyhow::Result<()> {
        let mut parser = Decoder::default();

        let preamble = b"\x00asm\x01\0\0\0";
        let custom = b"\x00\x04abcd";
        let types = b"\x01\x07\x02\x60\x02\x7f\x7f\x01\x7f\x60\0\x01\x7f";

        parser.write(preamble);
        parser.write(custom);
        parser.write(types);
        dbg!(parser.flush());

        Ok(())
    }

    #[test]
    fn parser2_works_mostly() -> anyhow::Result<()> {
        let mut parser = Decoder::default();

        dbg!(parser.write(include_bytes!("../test.wasm")));
        dbg!(parser.flush());
        Ok(())
    }

    //

    #[test]
    fn parser2_works_fr_fr() -> anyhow::Result<()> {
        let mut parser = Decoder::default();

        dbg!(parser.write(include_bytes!("../../../src/testsuite/func.0.wasm")));
        dbg!(parser.flush());
        Ok(())
    }
}
