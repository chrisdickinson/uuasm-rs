pub(crate) mod decoder;
pub(crate) mod original;
pub(crate) mod parser;
pub(crate) mod window;

#[inline]
#[cold]
pub(crate) fn cold() {}

pub use crate::original::parse as old_parse;

pub use crate::parser::any::AnyParser;
pub use crate::parser::module::ModuleParser;

pub use decoder::Decoder;

use thiserror::Error;
use uuasm_nodes::IR;
use window::{AdvancementError, DecodeWindow};

#[derive(Error, Debug, Clone)]
#[error(transparent)]
pub struct IRError<T: Clone + std::fmt::Debug + std::error::Error>(#[from] T);

#[derive(Error, Debug, Clone)]
#[error("{kind} (pos: {position})")]
pub struct ParseError<T: Clone + std::fmt::Debug + std::error::Error> {
    #[source]
    pub kind: ParseErrorKind<T>,
    pub position: usize,
}

#[derive(Error, Debug, Clone)]
pub enum ParseErrorKind<T: Clone + std::fmt::Debug + std::error::Error> {
    #[error("Bad magic number (expected 0061736DH ('\\0asm'), got {0:X}H")]
    BadMagic(u32),

    #[error("Bad type prefix (expected 60H, got {0:X}H)")]
    BadTypePrefix(u8),

    #[error("Bad type (got {0:X}H)")]
    BadType(u8),

    #[error("Bad import descriptor type (got {0:X}H)")]
    BadImportDesc(u8),

    #[error("Bad export descriptor type (got {0:X}H)")]
    BadExportDesc(u8),

    #[error("Bad data segment type (got {0:X}H; expected 0, 1, or 2)")]
    BadDataType(u8),

    #[error("Bad limit type (got {0:X}H; expected 0, or 1)")]
    BadLimitType(u8),

    #[error("Bad instruction (got {0:X}H)")]
    BadInstruction(u8),

    #[error(
        "malformed mutability value for global type definition (got {0:X}H; expected const=0 or mut=1)"
    )]
    BadMutability(u8),

    #[error("Unexpected version {0}")]
    UnexpectedVersion(u32),

    #[error("Unexpected end of stream")]
    UnexpectedEOS,

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
    Ready,
    YieldTo(AnyParser<T>, ResumeFunc<T>),
    YieldToBounded(u32, AnyParser<T>, ResumeFunc<T>),
}

#[allow(type_alias_bounds)]
pub type ResumeFunc<T: IR> =
    fn(&mut T, AnyParser<T>, AnyParser<T>) -> Result<AnyParser<T>, ParseErrorKind<T::Error>>;
#[allow(type_alias_bounds)]
pub type ParseResult<T: IR> = Result<Advancement<T>, ParseErrorKind<T::Error>>;

pub trait Parse<T: IR> {
    type Production: Sized;

    fn advance(&mut self, irgen: &mut T, window: &mut DecodeWindow) -> ParseResult<T>;
    fn production(self, irgen: &mut T) -> Result<Self::Production, ParseErrorKind<T::Error>>;
}

pub trait ExtractTarget<T>: Sized {
    fn extract(value: T) -> Result<Self, ExtractError>;
}

pub fn parse<T: IR>(irgen: T, input: &[u8]) -> Result<T::Module, ParseError<T::Error>> {
    let mut parser =
        Decoder::<T, T::Module>::new(AnyParser::Module(ModuleParser::default()), irgen);

    parser.write(input)?;
    parser.flush()
}
