# a tiny uuebassembly interpreter

Implementing WebAssembly in Rust for fun and education!

If you're looking for a industry-strength Wasm runtime, look at
[Wasmtime](https://github.com/bytecodealliance/wasmtime).

## PLAN

A semi-regularly updated dev log.

### 2024 Aug 17

I'm backfilling the dev log a bit here: let's talk a bit about this project and
its goals.

I have three concrete, personal goals in starting this project:

1. Create a webassembly project that doesn't start with "w" (hence the
   double-"u".)
2. Learn more about WebAssembly by building a runtime soup-to-nuts: a parser,
   interpreter, formatters, etc.
3. Learn more about how to structure interpreters in Rust.

My loosely-held hope is to develop this into a runtime that's focused on
developer debugging. But that's a ways off.

I have been focused on passing the WebAssembly test suite. The current thrust
is to implement WebAssembly validation so that I can restructure how the
runtime models WebAssembly values.

I made a simplifying assumption early on that I've found is no longer valid --
representing values naively using a Rust enum (`Value::F32(f32)`.) My thinking
is that implementing validation will allow me to make assertions about the
types on the interpreter stack so that I can represent them as variable-length
bytes. Flattening the representation this way will help us pass floating point
NaN canonicalization tests (and some of the hairier `ref.func` tests.)

I work on this project for about half an hour to an hour most nights; progress
is steady but by no means rapid.

Anyway, since I'm going to try to start working on this in the open, it's time
to do some spring cleaning:

- I should delete the old `src/` directory. I recently split up the parser, IR,
  and runtime sections into crates.
- Rename the `nodes` crate to `ir`, because A) it's short and B) it's more descriptive
- Move all of the `.wat` and `.wasm` that's committed to the repo into a `corpus` directory.
- Consider moving the tests into their own crate

### 2024 June 25

Still working on the items from yesterday (still! omg!)

Mid-breaking-up the crate into sub-crates there was a strange set of errors
that suggested that I had accidentally blown away my rust stdlib? but I think
that was illusory. Fixing the improper cross-crate `impl <struct>`'s fixed the
problem.

Other notes:

- I think "nodes" might become an "ir" crate.
    - here's the dumb ir idea: instructions are 32-bit; 12-16 bits are instr ids, the remaining 20-18 bits
      represent an index into a instruction arg table
        - I wanna make _room_ for this idea but not implement it yet because well, we still don't pass all
          tests yet, let alone having any benchmarks for the thing.
- the parser might keep state
    - some of that state might be an InternMap (and we might make Name take a Cow/cow-like-struct)
- editor stuff: pressing tab or jumping paragraphs sometimes sends me on a wild goose chase and it's driving me
  up a wall. it seems lsp-related, and in particular I've read that snippets might cause the problem.

---

### 2024 June 24

Up next:

- [x] fix the Take<P>::advance function (it currently faults on a subtraction)
- [x] finish type section
- [x] break up the workspace into crates: nodes, parser, runtime
- [x] break the parser up into files
- [x] target sub-productions to make it easier to write tests
- [ ] look into injecting parser middleware
- [ ] ask ourselves: is there an easy way to support "read me a vec of this type"?

---

## License

Apache 2.0
