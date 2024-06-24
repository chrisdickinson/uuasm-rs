use std::{cmp::min, marker::PhantomData, ops::RangeTo};

use thiserror::Error;

use crate::nodes::{Module, ModuleBuilder, SectionType, Type};

#[derive(Error, Debug, Clone)]
pub enum ParseError {
    #[error("incomplete stream: {0} bytes")]
    Incomplete(usize),

    #[error("unexpected end of stream: expected {0} bytes")]
    Expected(usize),

    #[error("Bad magic number (expected 0061736DH ('\\0asm'), got {0:x}")]
    BadMagic(u32),

    #[error("Unexpected version {0}")]
    UnexpectedVersion(u32),

    #[error("invalid section type {kind} at position {position}")]
    SectionInvalid { kind: u8, position: usize },

    #[error("invalid parser state: {0}")]
    InvalidState(&'static str),
}

enum Advancement {
    Ready(usize),
    YieldTo(usize, ParseState, ResumeFunc),
}

type ResumeFunc = fn(ParseState, ParseState) -> Result<ParseState, ParseError>;
type ParseResult = Result<Advancement, ParseError>;

trait Parse {
    type Production: Sized;

    fn advance(&mut self, window: ParserWindow) -> ParseResult;
    fn production(self) -> Result<Self::Production, ParseError>;
}

impl<T: num::Integer + Default> LEBParser<T> {
    fn new() -> Self {
        Self {
            repr: Default::default(),
            offs: 0,
            _marker: PhantomData,
        }
    }
}

impl<T: LEBConstants> Parse for LEBParser<T> {
    type Production = T;

    fn advance(&mut self, mut window: ParserWindow) -> ParseResult {
        let mut next;
        let mut shift = self.offs * 7;
        while {
            next = window.peek()?;

            self.repr |= ((next & 0x7f) as u64) << shift;
            shift += 7;

            next & 0x80 != 0
        } {
            window.take().unwrap();
        }
        window.take().unwrap();

        Ok(Advancement::Ready(window.offset()))
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        Ok(T::from_u64(self.repr))
    }
}

enum ParseState {
    LEBI32(LEBParser<i32>),
    LEBI64(LEBParser<i64>),
    LEBU32(LEBParser<u32>),
    LEBU64(LEBParser<u64>),
    Type(TypeParser),
    TypeSequence(TypeSequence),
    Accumulate(Take<Accumulator>),
    Section(SectionParser),
    Module(ModuleParser),

    Failed(ParseError),
}

impl Parse for ParseState {
    type Production = Module;

    fn advance(&mut self, window: ParserWindow<'_>) -> ParseResult {
        match self {
            ParseState::Failed(e) => Err(e.clone()),
            ParseState::LEBI32(p) => p.advance(window),
            ParseState::LEBI64(p) => p.advance(window),
            ParseState::LEBU32(p) => p.advance(window),
            ParseState::LEBU64(p) => p.advance(window),
            ParseState::Type(p) => p.advance(window),
            ParseState::TypeSequence(p) => p.advance(window),
            ParseState::Accumulate(p) => p.advance(window),
            ParseState::Section(p) => p.advance(window),
            ParseState::Module(p) => p.advance(window),
        }
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        let Self::Module(module) = self else {
            unreachable!();
        };

        module.production()
    }
}

struct Take<P: Parse> {
    inner: P,
    offset: usize,
    limit: usize,
}

impl<P: Parse> Take<P> {
    fn new(parser: P, limit: usize) -> Self {
        Self {
            inner: parser,
            offset: 0,
            limit,
        }
    }
}

trait LEBConstants {
    const MAX_BYTES: usize;
    const SIGNED: bool = false;
    fn from_u64(i: u64) -> Self;
}

impl LEBConstants for u32 {
    const MAX_BYTES: usize = 5;
    #[inline]
    fn from_u64(i: u64) -> Self {
        i as u32
    }
}

impl LEBConstants for u64 {
    const MAX_BYTES: usize = 10;
    #[inline]
    fn from_u64(i: u64) -> Self {
        i
    }
}

impl LEBConstants for i32 {
    const MAX_BYTES: usize = 5;
    const SIGNED: bool = true;
    #[inline]
    fn from_u64(i: u64) -> Self {
        i as i32
    }
}

impl LEBConstants for i64 {
    const MAX_BYTES: usize = 10;
    const SIGNED: bool = true;
    #[inline]
    fn from_u64(i: u64) -> Self {
        i as i64
    }
}

#[derive(Default)]
struct LEBParser<T> {
    repr: u64,
    offs: usize,
    _marker: PhantomData<T>,
}

#[derive(Default)]
enum SectionParser {
    #[default]
    ParseType,
    ParseLength(u8),
    SectionContent(u8, u32),
    Done(SectionType),
}

#[derive(Default)]
enum ModuleParser {
    #[default]
    Magic,
    TakeSection(Box<Option<ModuleBuilder>>),
    Done(Box<Module>),
}

impl Parse for ModuleParser {
    type Production = Module;

    fn advance(&mut self, window: ParserWindow) -> ParseResult {
        match self {
            ModuleParser::Magic => Ok(Advancement::YieldTo(
                window.offset(),
                ParseState::Accumulate(Take::new(Accumulator::new(8), 8)),
                |last_state, this_state| {
                    let ParseState::Accumulate(accum) = last_state else {
                        unreachable!()
                    };
                    let ParseState::Module(_) = this_state else {
                        unreachable!()
                    };
                    let production = accum.production()?;

                    let magic = &production[0..4];
                    if magic != b"\x00asm" {
                        return Err(ParseError::BadMagic(u32::from_ne_bytes(
                            magic.try_into().unwrap(),
                        )));
                    }

                    let version = &production[4..];
                    if version != b"\x01\x00\x00\x00" {
                        return Err(ParseError::UnexpectedVersion(u32::from_le_bytes(
                            magic.try_into().unwrap(),
                        )));
                    }

                    Ok(ParseState::Module(ModuleParser::TakeSection(Box::new(
                        Some(ModuleBuilder::new()),
                    ))))
                },
            )),

            ModuleParser::TakeSection(builder) => {
                match window.peek() {
                    Err(ParseError::Expected(1)) => {
                        let builder = builder.take().unwrap();
                        *self = ModuleParser::Done(Box::new(builder.build()));
                        return Ok(Advancement::Ready(window.offset()));
                    }
                    Err(err) => return Err(err),
                    _ => {}
                }

                Ok(Advancement::YieldTo(
                    window.offset(),
                    ParseState::Section(Default::default()),
                    |last_state, this_state| {
                        let ParseState::Section(section) = last_state else {
                            unreachable!();
                        };
                        let ParseState::Module(ModuleParser::TakeSection(mut builder_box)) =
                            this_state
                        else {
                            unreachable!();
                        };

                        let builder = builder_box.take().unwrap();
                        let section_type = section.production()?;
                        builder_box.replace(match section_type {
                            SectionType::Custom(xs) => builder.custom_section(xs),
                            SectionType::Type(xs) => builder.type_section(xs),
                            SectionType::Import(_) => todo!(),
                            SectionType::Function(_) => todo!(),
                            SectionType::Table(_) => todo!(),
                            SectionType::Memory(_) => todo!(),
                            SectionType::Global(_) => todo!(),
                            SectionType::Export(_) => todo!(),
                            SectionType::Start(_) => todo!(),
                            SectionType::Element(_) => todo!(),
                            SectionType::Code(_) => todo!(),
                            SectionType::Data(_) => todo!(),
                            SectionType::DataCount(_) => todo!(),
                        });

                        Ok(ParseState::Module(ModuleParser::TakeSection(builder_box)))
                    },
                ))
            }

            ModuleParser::Done(_) => Ok(Advancement::Ready(window.offset())),
        }
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        let ModuleParser::Done(module) = self else {
            unreachable!();
        };

        Ok(*module)
    }
}

impl SectionParser {
    fn resume_after_section_body(
        last_state: ParseState,
        this_state: ParseState,
    ) -> Result<ParseState, ParseError> {
        Ok(match (this_state, last_state) {
            (
                ParseState::Section(SectionParser::SectionContent(0x0, _)),
                ParseState::Accumulate(acc @ Take { .. }),
            ) => ParseState::Section(SectionParser::Done(SectionType::Custom(acc.production()?))),

            (
                ParseState::Section(SectionParser::SectionContent(0x1, _)),
                ParseState::TypeSequence(TypeSequence(ts)),
            ) => ParseState::Section(SectionParser::Done(SectionType::Type(ts.production()?))),
            _ => unreachable!(),
        })
    }
}

impl Parse for SectionParser {
    type Production = SectionType;

    fn advance(&mut self, mut window: ParserWindow) -> ParseResult {
        loop {
            match self {
                SectionParser::ParseType => {
                    *self = SectionParser::ParseLength(window.take()?);
                }
                SectionParser::ParseLength(xs) => {
                    return Ok(Advancement::YieldTo(
                        window.offset(),
                        ParseState::LEBU32(LEBParser::new()),
                        |last_state, this_state| {
                            let ParseState::LEBU32(leb) = last_state else {
                                unreachable!()
                            };
                            let ParseState::Section(SectionParser::ParseLength(kind)) = this_state
                            else {
                                unreachable!()
                            };

                            let len = leb.production()?;
                            Ok(ParseState::Section(SectionParser::SectionContent(
                                kind, len,
                            )))
                        },
                    ));
                }

                SectionParser::SectionContent(kind, length) => {
                    let length = *length as usize;
                    return Ok(match *kind {
                        0x0 => Advancement::YieldTo(
                            window.offset(),
                            ParseState::Accumulate(Take::new(Accumulator::new(length), length)),
                            SectionParser::resume_after_section_body,
                        ),

                        0x1 => Advancement::YieldTo(
                            window.offset(),
                            ParseState::TypeSequence(TypeSequence::new()),
                            SectionParser::resume_after_section_body,
                        ),

                        // 0x1 => SectionType::Type(Vec::<Type>::from_wasm_bytes(section)?.1),
                        unk => {
                            return Err(ParseError::SectionInvalid {
                                kind: unk,
                                position: window.position(),
                            })
                        }
                    });
                }
                SectionParser::Done(_) => return Ok(Advancement::Ready(window.offset())),
            };
        }
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        let Self::Done(section_type) = self else {
            unreachable!();
        };

        Ok(section_type)
    }
}

impl<P: Parse> Parse for Take<P> {
    type Production = P::Production;

    fn advance(&mut self, mut window: ParserWindow) -> ParseResult {
        if self.offset + window.available() > self.limit {
            let offset = window.offset();
            window = window.slice(..(self.limit - offset));
        }
        self.inner.advance(window)
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        self.inner.production()
    }
}

#[derive(Default)]
struct TypeParser;

impl Parse for TypeParser {
    type Production = Type;

    fn advance(&mut self, _window: ParserWindow) -> ParseResult {
        todo!()
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        todo!()
    }
}

struct TypeSequence(Sequence<TypeParser>);

impl TypeSequence {
    fn new() -> Self {
        Self(Sequence::Init(
            |lhs, rhs| {
                let ParseState::TypeSequence(xs) = rhs else {
                    unreachable!()
                };
                Ok(ParseState::TypeSequence(TypeSequence(
                    Sequence::<TypeParser>::resume_after_parse_len(lhs, xs.0)?,
                )))
            },
            |lhs, rhs| {
                let ParseState::Type(lhs) = lhs else {
                    unreachable!()
                };
                let ParseState::TypeSequence(rhs) = rhs else {
                    unreachable!()
                };
                Ok(ParseState::TypeSequence(TypeSequence(Sequence::<
                    TypeParser,
                >::resume_after_item(
                    lhs, rhs.0
                )?)))
            },
        ))
    }
}

impl Parse for TypeSequence {
    type Production = <Sequence<TypeParser> as Parse>::Production;

    fn advance(&mut self, window: ParserWindow) -> ParseResult {
        self.0.advance(window)
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        self.0.production()
    }
}

impl From<TypeParser> for ParseState {
    fn from(value: TypeParser) -> Self {
        ParseState::Type(value)
    }
}

enum Sequence<P: Parse> {
    Init(ResumeFunc, ResumeFunc),
    Allocated(Vec<P::Production>, usize, ResumeFunc),
    Ready(Box<[P::Production]>),
}

impl<P: Parse> Sequence<P> {
    fn resume_after_parse_len(
        last_state: ParseState,
        this_seq: Sequence<P>,
    ) -> Result<Self, ParseError> {
        let ParseState::LEBU32(leb) = last_state else {
            unreachable!();
        };
        let count = leb.production()? as usize;
        let Sequence::Init(_, resume_on_item) = this_seq else {
            unreachable!();
        };
        Ok(Sequence::Allocated(
            Vec::with_capacity(count),
            count,
            resume_on_item,
        ))
    }

    fn resume_after_item(last_parser: P, this_seq: Sequence<P>) -> Result<Self, ParseError> {
        let Sequence::Allocated(mut v, target, resume_on_item) = this_seq else {
            unreachable!();
        };
        v.push(last_parser.production()?);
        Ok(if v.len() == target {
            Sequence::Ready(v.into())
        } else {
            Sequence::Allocated(v, target, resume_on_item)
        })
    }
}

impl<P: Parse + Default + Into<ParseState>> Parse for Sequence<P> {
    type Production = Box<[P::Production]>;

    fn advance(&mut self, window: ParserWindow) -> ParseResult {
        match self {
            Sequence::Init(resume_after_len, _) => Ok(Advancement::YieldTo(
                window.offset(),
                ParseState::LEBU32(LEBParser::new()),
                *resume_after_len,
            )),
            Sequence::Allocated(_, _, resume_after_item) => Ok(Advancement::YieldTo(
                window.offset(),
                P::default().into(),
                *resume_after_item,
            )),
            Sequence::Ready(_) => todo!(),
        }
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        let Sequence::Ready(result) = self else {
            unreachable!();
        };
        Ok(result)
    }
}

struct Accumulator(usize, Box<[u8]>);

impl Accumulator {
    fn new(expected: usize) -> Self {
        Accumulator(0, vec![0; expected].into())
    }
}

impl Parse for Accumulator {
    type Production = Box<[u8]>;

    fn advance(&mut self, mut window: ParserWindow) -> ParseResult {
        let into = &mut self.1[self.0..];
        if !into.is_empty() {
            self.0 += window.take_n(into)?;
        }
        if self.0 == self.1.len() {
            Ok(Advancement::Ready(window.offset()))
        } else {
            Err(ParseError::Incomplete(self.1.len() - self.0))
        }
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        Ok(self.1)
    }
}

/*
 * A parser has a number of states, held in a stack. The last item in the stack is the current
 * parser. The parser can be written to, or flushed. When new bytes come in, the topmost stack
 * item processes them. If it is complete, it returns control to the next stack state along with
 * the production. The stack state can `take(N)`, `peek(N)`, `skip(N)`.
 */
pub struct Parser {
    state: Vec<(ParseState, ResumeFunc)>,
    position: usize,
}

fn noop_resume(_last_state: ParseState, _this_state: ParseState) -> Result<ParseState, ParseError> {
    Err(ParseError::InvalidState("this state should be unreachable"))
}

impl Default for Parser {
    fn default() -> Self {
        let mut state = Vec::with_capacity(16);
        state.push((
            ParseState::Module(ModuleParser::default()),
            noop_resume as ResumeFunc,
        ));
        Self { state, position: 0 }
    }
}

impl Parser {
    pub fn new() -> Self {
        Default::default()
    }

    fn write_inner<'a>(
        &'_ mut self,
        chunk: &'a [u8],
        eos: bool,
    ) -> Result<(Module, &'a [u8]), ParseError> {
        let mut window = ParserWindow {
            chunk,
            offset: 0,
            start_pos: self.position,
            eos,
        };
        loop {
            let (mut state, resume) = self.state.pop().unwrap();
            match state.advance(window) {
                Ok(Advancement::Ready(offset)) => {
                    if self.state.is_empty() {
                        let module = state.production()?;
                        return Ok((module, &chunk[offset..]));
                    }

                    let (receiver, last_resume) = self.state.pop().unwrap();
                    self.state.push((resume(state, receiver)?, last_resume));
                    window = ParserWindow {
                        chunk,
                        offset,
                        start_pos: self.position,
                        eos: false,
                    };
                }

                Ok(Advancement::YieldTo(offset, next_state, next_resume)) => {
                    self.state.push((state, resume));
                    self.state.push((next_state, next_resume));
                    window = ParserWindow {
                        chunk,
                        offset,
                        start_pos: self.position,
                        eos: false,
                    };
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
            }
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

#[derive(Debug)]
struct ParserWindow<'a> {
    chunk: &'a [u8],
    offset: usize,
    start_pos: usize,
    eos: bool,
}

impl<'a> ParserWindow<'a> {
    fn available(&self) -> usize {
        self.chunk.len() - self.offset
    }

    /// Offset represents the number of bytes consumed from the current chunk.
    fn offset(&self) -> usize {
        self.offset
    }

    /// Position represents the number of bytes consumed from the entire stream.
    fn position(&self) -> usize {
        self.offset + self.start_pos
    }

    fn slice(self, range: RangeTo<usize>) -> Self {
        Self {
            chunk: &self.chunk[..range.end],
            offset: self.offset,
            start_pos: self.start_pos,
            eos: true,
        }
    }

    fn take(&mut self) -> Result<u8, ParseError> {
        let next = self.peek()?;
        self.offset += 1;
        Ok(next)
    }

    fn take_n(&mut self, into: &mut [u8]) -> Result<usize, ParseError> {
        let dstlen = into.len();

        let src = &self.chunk[self.offset..];
        if src.is_empty() {
            return Err(if self.eos {
                ParseError::Expected(dstlen)
            } else {
                ParseError::Incomplete(dstlen)
            });
        }

        let to_write = min(dstlen, src.len());
        into[0..to_write].copy_from_slice(&src[0..to_write]);
        self.offset += to_write;
        Ok(to_write)
    }

    fn peek(&self) -> Result<u8, ParseError> {
        if self.offset >= self.chunk.len() {
            Err(if self.eos {
                ParseError::Expected(1)
            } else {
                ParseError::Incomplete(1)
            })
        } else {
            Ok(self.chunk[self.offset])
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parser2_works_basically() -> anyhow::Result<()> {
        let mut parser = Parser::default();

        let preamble = b"\x00asm\x01\0\0\0";
        let custom = b"\x00\x04abcd";

        dbg!(parser.write(preamble));
        dbg!(parser.write(custom));
        dbg!(parser.flush());

        Ok(())
    }
}
