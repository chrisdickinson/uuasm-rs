#![allow(dead_code)]
use std::marker::PhantomData;

use uuasm_nodes::IR;

use crate::{window::DecodeWindow, Parse, ParseError, ParseResult};

pub struct Take<T: IR, P: Parse<T>> {
    inner: P,
    offset: usize,
    limit: usize,
    _marker: PhantomData<T>,
}

impl<T: IR, P: Parse<T>> Take<T, P> {
    pub(crate) fn new(parser: P, limit: usize) -> Self {
        Self {
            inner: parser,
            offset: 0,
            limit,
            _marker: PhantomData,
        }
    }

    pub(crate) fn map<F: FnOnce(P) -> Result<P, ParseError<T::Error>>>(
        self,
        mapper: F,
    ) -> Result<Self, ParseError<T::Error>> {
        let Self {
            inner,
            offset,
            limit,
            _marker,
        } = self;

        Ok(Self {
            inner: mapper(inner)?,
            offset,
            limit,
            _marker,
        })
    }
}

impl<T: IR, P: Parse<T>> Parse<T> for Take<T, P> {
    type Production = P::Production;

    fn advance(&mut self, irgen: &mut T, mut window: DecodeWindow) -> ParseResult<T> {
        if self.offset + window.available() >= self.limit {
            window = window.slice(self.limit - self.offset);
            self.offset = self.limit;
        } else {
            self.offset += window.available();
        }
        self.inner.advance(irgen, window)
    }

    fn production(self, irgen: &mut T) -> Result<Self::Production, ParseError<T::Error>> {
        self.inner.production(irgen)
    }
}
