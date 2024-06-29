pub(crate) mod decoder;
pub(crate) mod parser;
pub(crate) mod window;

use crate::parser::state::ParseState;

pub use decoder::Decoder;

use thiserror::Error;
use window::DecodeWindow;

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

    fn advance(&mut self, window: DecodeWindow) -> ParseResult;
    fn production(self) -> Result<Self::Production, ParseError>;
}
