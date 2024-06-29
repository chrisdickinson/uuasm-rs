use uuasm_nodes::Module;

use crate::{
    parser::{module::ModuleParser, state::ParseState},
    window::DecodeWindow,
    Advancement, Parse, ParseError, ResumeFunc,
};

/*
 * A parser has a number of states, held in a stack. The last item in the stack is the current
 * parser. The parser can be written to, or flushed. When new bytes come in, the topmost stack
 * item processes them. If it is complete, it returns control to the next stack state along with
 * the production. The stack state can `take(N)`, `peek(N)`, `skip(N)`.
 */
pub struct Decoder {
    state: Vec<(ParseState, ResumeFunc)>,
    position: usize,
}

fn noop_resume(_last_state: ParseState, _this_state: ParseState) -> Result<ParseState, ParseError> {
    Err(ParseError::InvalidState("this state should be unreachable"))
}

impl Default for Decoder {
    fn default() -> Self {
        let mut state = Vec::with_capacity(16);
        state.push((
            ParseState::Module(ModuleParser::default()),
            noop_resume as ResumeFunc,
        ));
        Self { state, position: 0 }
    }
}

impl Decoder {
    pub fn new() -> Self {
        Default::default()
    }

    fn write_inner<'a>(
        &'_ mut self,
        chunk: &'a [u8],
        eos: bool,
    ) -> Result<(Module, &'a [u8]), ParseError> {
        let mut window = DecodeWindow::new(chunk, 0, self.position, eos);
        loop {
            let (mut state, resume) = self.state.pop().unwrap();
            let offset = match state.advance(window) {
                Ok(Advancement::Ready(offset)) => {
                    if self.state.is_empty() {
                        let module = state.production()?;
                        return Ok((module, &chunk[offset..]));
                    }

                    let (receiver, last_resume) = self.state.pop().unwrap();
                    self.state.push((resume(state, receiver)?, last_resume));
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
                        .push((ParseState::Failed(e.clone()), noop_resume as ResumeFunc));
                    return Err(e);
                }
            };
            window = DecodeWindow::new(chunk, offset, self.position, eos);
        }
    }

    pub fn write<'a>(&'_ mut self, chunk: &'a [u8]) -> Result<(Module, &'a [u8]), ParseError> {
        self.write_inner(chunk, false)
    }

    pub fn flush(&mut self) -> Result<Module, ParseError> {
        let (module, _) = self.write_inner(&[], true)?;
        Ok(module)
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
