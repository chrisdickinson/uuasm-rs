use crate::{window::DecodeWindow, Parse, ParseError, ParseResult};

pub struct Take<P: Parse> {
    inner: P,
    offset: usize,
    limit: usize,
}

impl<P: Parse> Take<P> {
    pub(crate) fn new(parser: P, limit: usize) -> Self {
        Self {
            inner: parser,
            offset: 0,
            limit,
        }
    }

    pub(crate) fn map<F: FnOnce(P) -> Result<P, ParseError>>(
        self,
        mapper: F,
    ) -> Result<Self, ParseError> {
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

    fn advance(&mut self, mut window: DecodeWindow) -> ParseResult {
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
