pub(crate) mod decoder;
pub(crate) mod parser;
pub(crate) mod window;

use std::str::Utf8Error;

use crate::parser::any::AnyParser;

pub use decoder::Decoder;

use thiserror::Error;
use uuasm_nodes::IR;
use window::{AdvancementError, DecodeWindow};

#[derive(Error, Debug, Clone)]
#[error(transparent)]
pub struct IRError<T: Clone + std::fmt::Debug + std::error::Error>(#[from] T);

#[derive(Error, Debug, Clone)]
pub enum ParseError<T: Clone + std::fmt::Debug + std::error::Error> {
    #[error("Bad magic number (expected 0061736DH ('\\0asm'), got {0:X}H")]
    BadMagic(u32),

    #[error("Bad type prefix (expected 60H, got {0:X}H)")]
    BadTypePrefix(u8),

    #[error("Bad type (got {0:X}H)")]
    BadType(u8),

    #[error("Bad import descriptor type (got {0:X}H)")]
    BadImportDesc(u8),

    #[error("Unexpected version {0}")]
    UnexpectedVersion(u32),

    #[error("invalid section type {kind} at position {position}")]
    SectionInvalid { kind: u8, position: usize },

    #[error("invalid parser state: {0}")]
    InvalidState(&'static str),

    #[error("Invalid import descriptor: {0}")]
    InvalidImportDescriptor(u8),

    #[error(transparent)]
    InvalidProduction(#[from] ExtractError),

    #[error(transparent)]
    Advancement(#[from] AdvancementError),

    #[error("IR error: {0}")]
    IRError(#[from] IRError<T>),
}

#[derive(Error, Debug, Clone)]
pub enum ExtractError {
    #[error("Failed to extract")]
    Failed,
}

pub enum Advancement<T: IR> {
    Ready(usize),
    YieldTo(usize, AnyParser<T>, ResumeFunc<T>),
}

pub type ResumeFunc<T: IR> =
    fn(&mut T, AnyParser<T>, AnyParser<T>) -> Result<AnyParser<T>, ParseError<T::Error>>;
pub type ParseResult<T: IR> = Result<Advancement<T>, ParseError<T::Error>>;

pub trait Parse<T: IR> {
    type Production: Sized;

    fn advance(&mut self, irgen: &mut T, window: DecodeWindow) -> ParseResult<T>;
    fn production(self, irgen: &mut T) -> Result<Self::Production, ParseError<T::Error>>;
}

pub trait ExtractTarget<T>: Sized {
    fn extract(value: T) -> Result<Self, ExtractError>;
}
