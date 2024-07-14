use std::cmp::min;

use thiserror::Error;

#[derive(Error, Debug, Clone, Copy)]
pub enum AdvancementError {
    #[error("incomplete stream: {0} bytes at position {1}")]
    Expected(usize, usize),
    #[error("unexpected end of stream: expected {0} bytes")]
    Incomplete(usize),
}

#[derive(Debug)]
pub struct DecodeWindow<'a> {
    chunk: &'a [u8],
    offset: usize,
    start_pos: usize,
    eos: bool,
}

impl<'a> DecodeWindow<'a> {
    pub fn new(
        chunk: &'a [u8],
        offset: usize,
        start_pos: usize,
        eos: bool,
        bound: Option<u32>,
    ) -> Self {
        let (eos, chunk) = if let Some(bound) = bound {
            let bound = bound as usize;
            (
                eos || chunk.len() > bound,
                &chunk[0..min(chunk.len(), bound)],
            )
        } else {
            (eos, chunk)
        };

        Self {
            chunk,
            offset,
            start_pos,
            eos,
        }
    }

    /// Offset represents the number of bytes consumed from the current chunk.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Position represents the number of bytes consumed from the entire stream.
    pub fn position(&self) -> usize {
        self.offset + self.start_pos
    }

    pub fn take(&mut self) -> Result<u8, AdvancementError> {
        let next = self.peek()?;
        self.offset += 1;
        Ok(next)
    }

    pub fn take_n(&mut self, into: &mut [u8]) -> Result<usize, AdvancementError> {
        let dstlen = into.len();

        let src = &self.chunk[self.offset..];
        if src.is_empty() {
            return Err(if self.eos {
                AdvancementError::Expected(dstlen, self.position())
            } else {
                AdvancementError::Incomplete(dstlen)
            });
        }

        let to_write = min(dstlen, src.len());
        into[0..to_write].copy_from_slice(&src[0..to_write]);
        self.offset += to_write;
        Ok(to_write)
    }

    pub fn peek(&self) -> Result<u8, AdvancementError> {
        if self.offset >= self.chunk.len() {
            Err(if self.eos {
                AdvancementError::Expected(1, self.position())
            } else {
                AdvancementError::Incomplete(1)
            })
        } else {
            Ok(self.chunk[self.offset])
        }
    }
}
