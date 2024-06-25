use std::{cmp::min, marker::PhantomData, mem, ops::RangeTo};

use thiserror::Error;

use crate::nodes::{
    Module, ModuleBuilder, NumType, RefType, ResultType, SectionType, Type, ValType, VecType,
};

#[derive(Error, Debug, Clone)]
pub enum ParseError {
    #[error("incomplete stream: {0} bytes")]
    Incomplete(usize),

    #[error("unexpected end of stream: expected {0} bytes")]
    Expected(usize),

    #[error("Bad magic number (expected 0061736DH ('\\0asm'), got {0:X}H")]
    BadMagic(u32),

    #[error("Bad type prefix (expected 60H, got {0:X}H)")]
    BadTypePrefix(u8),

    #[error("Bad type (got {0:X}H)")]
    BadType(u8),

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
    TypeSection(Take<TypeParser>),
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
            ParseState::TypeSection(p) => p.advance(window),
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
                ParseState::TypeSection(ts),
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
                SectionParser::ParseLength(_xs) => {
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
                            ParseState::TypeSection(Take::new(TypeParser::default(), length)),
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

enum TypeParser {
    Init(Vec<Type>),
    InputSize(Vec<Type>, u32),
    Input(Vec<Type>, Option<ResultType>),
    OutputSize(Vec<Type>, Option<ResultType>, u32),
    Output(Vec<Type>, Option<ResultType>, Option<ResultType>),
}

impl Default for TypeParser {
    fn default() -> Self {
        Self::Init(vec![])
    }
}

impl TypeParser {
    fn map_buffer_to_result_type(input_buf: Box<[u8]>) -> Result<ResultType, ParseError> {
        let mut types = Vec::with_capacity(input_buf.len());
        for item in &*input_buf {
            types.push(match item {
                0x6f => ValType::RefType(RefType::ExternRef),
                0x70 => ValType::RefType(RefType::FuncRef),
                0x7b => ValType::VecType(VecType::V128),
                0x7c => ValType::NumType(NumType::F64),
                0x7d => ValType::NumType(NumType::F32),
                0x7e => ValType::NumType(NumType::I64),
                0x7f => ValType::NumType(NumType::I32),
                byte => return Err(ParseError::BadType(*byte)),
            })
        }
        Ok(ResultType(types.into()))
    }
}

impl Parse for TypeParser {
    type Production = Box<[Type]>;

    fn advance(&mut self, mut window: ParserWindow) -> ParseResult {
        loop {
            *self = match self {
                TypeParser::Init(_) => {
                    match window.peek() {
                        Err(ParseError::Expected(1)) => {
                            return Ok(Advancement::Ready(window.offset()));
                        }
                        Err(err) => return Err(err),
                        _ => {}
                    }

                    let tag = window.take()?;
                    if tag != 0x60 {
                        return Err(ParseError::BadTypePrefix(tag));
                    }

                    return Ok(Advancement::YieldTo(
                        window.offset(),
                        ParseState::LEBU32(LEBParser::default()),
                        |last_state, this_state| {
                            let ParseState::LEBU32(leb) = last_state else {
                                unreachable!();
                            };

                            let ParseState::TypeSection(take) = this_state else {
                                unreachable!();
                            };

                            let entry_count = leb.production()?;
                            Ok(ParseState::TypeSection(take.map(|this_state| {
                                let Self::Init(v) = this_state else {
                                    unreachable!();
                                };
                                Ok(if entry_count == 0 {
                                    Self::Input(v, None)
                                } else {
                                    Self::InputSize(v, entry_count)
                                })
                            })?))
                        },
                    ));
                }
                TypeParser::InputSize(_, size) => {
                    let size = *size as usize;

                    return Ok(Advancement::YieldTo(
                        window.offset(),
                        ParseState::Accumulate(Take::new(Accumulator::new(size), size)),
                        |last_state, this_state| {
                            let ParseState::Accumulate(accum) = last_state else {
                                unreachable!()
                            };
                            let ParseState::TypeSection(take) = this_state else {
                                unreachable!();
                            };

                            let input_buf = accum.production()?;
                            let result_type = TypeParser::map_buffer_to_result_type(input_buf)?;
                            Ok(ParseState::TypeSection(take.map(|this_state| {
                                let Self::InputSize(v, _) = this_state else {
                                    unreachable!();
                                };
                                Ok(Self::Input(v, Some(result_type)))
                            })?))
                        },
                    ));
                }

                TypeParser::Input(_, _) => {
                    return Ok(Advancement::YieldTo(
                        window.offset(),
                        ParseState::LEBU32(LEBParser::default()),
                        |last_state, this_state| {
                            let ParseState::LEBU32(leb) = last_state else {
                                unreachable!();
                            };

                            let ParseState::TypeSection(take) = this_state else {
                                unreachable!();
                            };

                            let entry_count = leb.production()?;

                            Ok(ParseState::TypeSection(take.map(|this_state| {
                                let Self::Input(v, result_type) = this_state else {
                                    unreachable!();
                                };
                                Ok(if entry_count == 0 {
                                    Self::Output(v, result_type, None)
                                } else {
                                    Self::OutputSize(v, result_type, entry_count)
                                })
                            })?))
                        },
                    ));
                }
                TypeParser::OutputSize(_, _, size) => {
                    let size = *size as usize;

                    return Ok(Advancement::YieldTo(
                        window.offset(),
                        ParseState::Accumulate(Take::new(Accumulator::new(size), size)),
                        |last_state, this_state| {
                            let ParseState::Accumulate(accum) = last_state else {
                                unreachable!()
                            };
                            let ParseState::TypeSection(take) = this_state else {
                                unreachable!();
                            };

                            let output_buf = accum.production()?;
                            let result_type = TypeParser::map_buffer_to_result_type(output_buf)?;

                            Ok(ParseState::TypeSection(take.map(|this_state| {
                                let Self::OutputSize(v, input_result_type, _) = this_state else {
                                    unreachable!();
                                };
                                Ok(Self::Output(v, input_result_type, Some(result_type)))
                            })?))
                        },
                    ));
                }
                TypeParser::Output(v, input_type, output_type) => {
                    v.push(Type(
                        input_type.take().unwrap_or_default(),
                        output_type.take().unwrap_or_default(),
                    ));
                    TypeParser::Init(mem::take(v))
                }
            }
        }
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        let Self::Init(v) = self else {
            unreachable!();
        };

        Ok(v.into_boxed_slice())
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

    fn map<F: FnOnce(P) -> Result<P, ParseError>>(self, mapper: F) -> Result<Self, ParseError> {
        let Self {
            inner,
            offset,
            limit,
        } = self;

        Ok(Self {
            inner: mapper(inner)?,
            offset,
            limit,
        })
    }
}

impl<P: Parse> Parse for Take<P> {
    type Production = P::Production;

    fn advance(&mut self, mut window: ParserWindow) -> ParseResult {
        if self.offset + window.available() >= self.limit {
            window = window.slice(self.limit - self.offset);
            self.offset = self.limit;
        } else {
            self.offset += window.available();
        }
        self.inner.advance(window)
    }

    fn production(self) -> Result<Self::Production, ParseError> {
        self.inner.production()
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
                        eos,
                    };
                }

                Ok(Advancement::YieldTo(offset, next_state, next_resume)) => {
                    self.state.push((state, resume));
                    self.state.push((next_state, next_resume));
                    window = ParserWindow {
                        chunk,
                        offset,
                        start_pos: self.position,
                        eos,
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

    fn slice(self, take: usize) -> Self {
        Self {
            chunk: &self.chunk[..self.offset + take],
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
        let types = b"\x01\x06\x60\x02\x7f\x7f\x01\x7f";

        parser.write(preamble);
        parser.write(custom);
        parser.write(types);
        dbg!(parser.flush());

        Ok(())
    }
}
