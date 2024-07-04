use std::marker::PhantomData;

use uuasm_nodes::{DefaultIRGenerator, IR};

use crate::{
    parser::{
        any::{AnyParser, AnyProduction},
        module::ModuleParser,
    },
    window::DecodeWindow,
    Advancement, ExtractTarget, Parse, ParseError, ResumeFunc,
};

/*
 * A parser has a number of states, held in a stack. The last item in the stack is the current
 * parser. The parser can be written to, or flushed. When new bytes come in, the topmost stack
 * item processes them. If it is complete, it returns control to the next stack state along with
 * the production. The stack state can `take(N)`, `peek(N)`, `skip(N)`.
 */
pub struct Decoder<T: IR, Target: ExtractTarget<AnyProduction<T>>> {
    state: Vec<(AnyParser<T>, ResumeFunc<T>)>,
    position: usize,
    irgen: T,
    _marker: PhantomData<Target>,
}

fn noop_resume<T: IR>(
    _irgen: &mut T,
    _last_state: AnyParser<T>,
    _this_state: AnyParser<T>,
) -> Result<AnyParser<T>, ParseError> {
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
        let mut state = Vec::with_capacity(16);
        state.push((parser, noop_resume as ResumeFunc<T>));
        Self {
            state,
            position: 0,
            irgen,
            _marker: PhantomData,
        }
    }

    fn write_inner<'a>(
        &'_ mut self,
        chunk: &'a [u8],
        eos: bool,
    ) -> Result<(Target, &'a [u8]), ParseError> {
        let mut window = DecodeWindow::new(chunk, 0, self.position, eos);
        loop {
            let (mut state, resume) = self.state.pop().unwrap();
            let offset = match state.advance(&mut self.irgen, window) {
                Ok(Advancement::Ready(offset)) => {
                    if self.state.is_empty() {
                        let output = Target::extract(state.production(&mut self.irgen)?)?;
                        return Ok((output, &chunk[offset..]));
                    }

                    let (receiver, last_resume) = self.state.pop().unwrap();
                    self.state
                        .push((resume(&mut self.irgen, state, receiver)?, last_resume));
                    offset
                }

                Ok(Advancement::YieldTo(offset, next_state, next_resume)) => {
                    self.state.push((state, resume));
                    self.state.push((next_state, next_resume));
                    offset
                }

                Err(err @ ParseError::Incomplete(_)) => {
                    self.position += chunk.len();
                    self.state.push((state, resume));
                    return Err(err);
                }

                Err(e) => {
                    self.state
                        .push((AnyParser::Failed(e.clone()), noop_resume as ResumeFunc<T>));
                    return Err(e);
                }
            };
            window = DecodeWindow::new(chunk, offset, self.position, eos);
        }
    }

    pub fn write<'a>(&'_ mut self, chunk: &'a [u8]) -> Result<(Target, &'a [u8]), ParseError> {
        self.write_inner(chunk, false)
    }

    pub fn flush(&mut self) -> Result<Target, ParseError> {
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
        let types = b"\x01\x06\x60\x02\x7f\x7f\x01\x7f";

        parser.write(preamble);
        parser.write(custom);
        parser.write(types);
        dbg!(parser.flush());

        Ok(())
    }
}
