pub(crate) mod decoder;
pub(crate) mod parser;
pub(crate) mod window;

use std::str::Utf8Error;

use crate::parser::any::AnyParser;

pub use decoder::Decoder;

use thiserror::Error;
use uuasm_nodes::IR;
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

    #[error("Invalid utf-8 in name")]
    InvalidUTF8(#[from] Utf8Error),

    #[error("Invalid import descriptor: {0}")]
    InvalidImportDescriptor(u8),

    #[error("invalid production")]
    InvalidProduction,
}

pub enum Advancement<T: IR> {
    Ready(usize),
    YieldTo(usize, AnyParser<T>, ResumeFunc<T>),
}

pub type ResumeFunc<T> = fn(&mut T, AnyParser<T>, AnyParser<T>) -> Result<AnyParser<T>, ParseError>;
pub type ParseResult<T> = Result<Advancement<T>, ParseError>;

pub trait Parse<T: IR> {
    type Production: Sized;

    fn advance(&mut self, irgen: &mut T, window: DecodeWindow) -> ParseResult<T>;
    fn production(self, irgen: &mut T) -> Result<Self::Production, ParseError>;
}

pub trait ExtractTarget<T>: Sized {
    fn extract(value: T) -> Result<Self, ParseError>;
}
