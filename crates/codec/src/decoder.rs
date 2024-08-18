use std::marker::PhantomData;

use uuasm_ir::{DefaultIRGenerator, IR};

use crate::{
    cold,
    parser::{
        any::{AnyParser, AnyProduction},
        module::ModuleParser,
    },
    window::{AdvancementError, DecodeWindow},
    Advancement, ExtractTarget, Parse, ParseError, ParseErrorKind, ResumeFunc,
};

/*
 * A parser has a number of states, held in a stack. The last item in the stack is the current
 * parser. The parser can be written to or flushed. When new bytes come in, the topmost stack
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
) -> Result<AnyParser<T>, ParseErrorKind<T::Error>> {
    Err(ParseErrorKind::InvalidState(
        "this state should be unreachable",
    ))
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
        let mut window = DecodeWindow::new(chunk, 0, self.position, eos, None);
        loop {
            #[cfg(any())]
            eprintln!(
                "{:x} {}",
                window.position(),
                self.state
                    .iter()
                    .map(|(parser, _, _)| format!("{parser:?}"))
                    .collect::<Vec<_>>()
                    .join("/")
            );
            // TODO: start checking bounds again
            let (state, resume, bound) = self.state.last_mut().unwrap();

            match state.advance(&mut self.irgen, &mut window) {
                Ok(Advancement::Ready) => {
                    let (_, _, _) = (state, resume, bound);
                    let (n_state, n_resume, _) = self.state.pop().unwrap();
                    let Some((receiver, last_resume, last_bound)) = self.state.pop() else {
                        cold();

                        let output = n_state
                            .production(&mut self.irgen)
                            .and_then(|xs| Target::extract(xs).map_err(Into::into))
                            .map_err(|kind| ParseError {
                                kind,
                                position: window.position(),
                            })?;

                        return Ok((output, &chunk[window.offset()..]));
                    };

                    let resumed = match n_resume(&mut self.irgen, n_state, receiver) {
                        Ok(v) => v,
                        Err(kind) => {
                            self.state.push((
                                AnyParser::Failed(kind.clone()),
                                noop_resume as ResumeFunc<T>,
                                None,
                            ));
                            return Err(ParseError {
                                kind,
                                position: window.position(),
                            });
                        }
                    };

                    self.state.push((resumed, last_resume, last_bound));
                }

                Ok(Advancement::YieldTo(next_state, next_resume)) => {
                    if let Some(xs) = bound.as_mut() {
                        *xs = xs.saturating_sub((window.position() - self.position) as u32);
                    }
                    let (_, _) = (state, resume);
                    let bound = *bound;
                    self.state.push((next_state, next_resume, bound));
                }

                Ok(Advancement::YieldToBounded(next_bound, next_state, next_resume)) => {
                    if let Some(xs) = bound.as_mut() {
                        *xs = xs.saturating_sub((window.position() - self.position) as u32);
                        if next_bound > *xs {
                            let kind = ParseErrorKind::UnexpectedEOS;
                            self.state.push((
                                AnyParser::Failed(kind.clone()),
                                noop_resume as ResumeFunc<T>,
                                None,
                            ));
                            return Err(ParseError {
                                kind,
                                position: window.position(),
                            });
                        }
                    }
                    self.state.push((next_state, next_resume, Some(next_bound)));
                }

                Err(err @ ParseErrorKind::Advancement(AdvancementError::Incomplete(_))) => {
                    if let Some(xs) = bound.as_mut() {
                        *xs = xs.saturating_sub((chunk.len() - window.offset()) as u32);
                    }
                    return Err(ParseError {
                        kind: err,
                        position: window.position(),
                    });
                }

                Err(e) => {
                    self.state.push((
                        AnyParser::Failed(e.clone()),
                        noop_resume as ResumeFunc<T>,
                        None,
                    ));
                    return Err(ParseError {
                        kind: e,
                        position: window.position(),
                    });
                }
            }
            self.position = window.position();
        }
    }

    pub fn write<'a>(
        &'_ mut self,
        chunk: &'a [u8],
    ) -> Result<Option<(Target, &'a [u8])>, ParseError<T::Error>> {
        match self.write_inner(chunk, false) {
            Ok(xs) => Ok(Some(xs)),
            Err(ParseError {
                kind: ParseErrorKind::Advancement(AdvancementError::Incomplete(_)),
                ..
            }) => Ok(None),
            Err(e) => Err(e),
        }
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
    fn parser2_works_mostly() -> anyhow::Result<()> {
        let mut parser = Decoder::default();

        let _ = dbg!(parser.write(include_bytes!("../test.wasm")));
        let _ = dbg!(parser.flush());
        Ok(())
    }

    //

    #[test]
    fn parser2_works_fr_fr() -> anyhow::Result<()> {
        let mut parser = Decoder::default();

        let _ = dbg!(parser.write(include_bytes!("../../rt/src/testsuite/func.0.wasm")));
        let _ = dbg!(parser.flush());
        Ok(())
    }
}
