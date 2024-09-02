# a tiny uuebassembly interpreter

Implementing WebAssembly in Rust for fun and education!

If you're looking for a industry-strength Wasm runtime, look at
[Wasmtime](https://github.com/bytecodealliance/wasmtime).

## PLAN

A semi-regularly updated dev log.

### 2024 Sep 01

Labor day weekend!

Loose focus:

- Finish fixing table tests.
- Make a separate "typechecker" IR generator that passes through to an internal IR generator
- Maybe fix the parser? I'm so tempted.

### 2024 Aug 30

["They say time is the fire in which we burn."](https://www.youtube.com/watch?v=XtIuC0NAF_E)

... which sums up how I'm feeling about getting the `if` opcode tests
to 100% passing!

So. If you're keeping score at home, I've been working on implementing type
checking in the `ir` sub-crate for a week or two now. I could have saved myself
some time at the outset: the WebAssembly spec has a useful appendix entry [which
describes][wasm-validate] how to implement the type checker. I spent about a day
or two at the outset trying to build my own checker from first principles before
stumbling on this appendix entry. The specification, unlike human anatomy, has
a pretty useful appendix.

In the process of implemeting type checks, I learned something new about
constant expressions: I knew that `global.get` was a valid "constant"
instruction, but I did *not* know that there were additional requirements on
the referent: in particular the target must be an immutable _import_. Neat.
(Module validation contains a lot of these little pearls of wisdom baked into
the specification. I feel like "This is actually my second rodeo" might be a
fitting motto for the specification.)

So what does the work look like right now? Well, a lot of `wasm-tools dump`,
`wasm-tools print`, and debug loglines. Since I'm spending a lot of time with
the parser and IR generator, I am (of course) starting to feel like the
abstractions aren't _quite_ right. The IR generator's type checker bleeds into
the default IR generation, the default IR is clunky, and the parser both
communicates with an IR generator (good!) and constructs final parsed
productions (bad!) However. Getting the tests to 100% is the first step, and
everything else is secondary.

In other news: in true crustacean fashion, Rust is getting underfoot. I'm
feeling the pinch.

As I mentioned, I have a default IR generator, called with a `&mut self`
reference. It owns definitions of the types, locals, globals, and tables. These
are built iteratively so several of them are behind `Option<Vec<[T]>>`-style
references. (Sometimes they're `Box<[T]>`.) Anyway, the type checker needs
access to this information when handling new instructions. As a result, I had a
lot of `self.<foo>_types.as_ref().map(|xs| xs as &[_]).unwrap_or_default()`
chains inlined as parameters to `type_checker.trace(instr)`. I thought I'd add
a method to get a `&[T]` record for these hairy types -- a little helper.

But you have to pay the crab tax for this. The borrow checker does a bunch of
work to make mutably borrowing one `self` attribute while immutably borrowing
adjacent `self` attributes work -- when inside of a single function. However,
they stop at method call boundaries, so helper functions that return references
make it impossible to _also_ use `self` mutably. There are [ways][view-types]
around this but they're pretty heavyweight right now. (See [more
recently][borrow-checker] on this.)

Kind of a long form update -- apologies -- but we're at 64 passing tests to 29
failing tests. The tide is turning! Soon we'll break all of the tests again
by changing the value stack!

[view-types]: https://smallcultfollowing.com/babysteps/blog/2021/11/05/view-types/
[borrow-checker]: https://smallcultfollowing.com/babysteps/blog/2024/06/02/the-borrow-checker-within/#step-3-view-types-and-interprocedural-borrows
[wasm-validate]: https://webassembly.github.io/spec/core/appendix/algorithm.html

### 2024 Aug 24

We're still knee-deep in module validation. In particular, type-checking
instructions. In exacting detail: we have to make the type checker work with
instruction sequences stored _outside_ of functions.

Where might instruction sequences be stored if not in functions? A number of
places, as it turns out! Wasm uses sequences of instructions to represent
element values (arrays of function pointers), data and element offsets, and
global values. To support these constructions, the IR generator needs hooks
that bookend these expressions --
`start_element_value_expr`/`end_element_value_expr`, etc.

This is a long, long yak shave: I started this process more than a month ago.
If I knew then what I know now: I'd try to drive more of the parser and
validator generation from a table data structure. Something stored in a
non-executable format (YAML? Ron? CSV? JSON?) Something that could be
processed by a series of macros.

It'd maybe store:

- An instruction "page" and "subcode" -- where subcode is 0 for all single-byte
  instructions.
- Mnemonics for the instruction: `i32.add`, for example. Maybe add a column for
  acceptable aliases?
- A sequence of instruction args, to be grouped up into `nullary`, `unary`,
  `binary`, etc.
- A little mini-language for pre-condition / post-condition types (`i32 ->
  i32`, `(i32, i32, i32) -> ()`, etc.)
- Maybe inlining some implementation? `i32.add` might include something like
  `%0 = %1 + %2`? Or some way to include an implementation defined elsewhere
  in code? (I'm [greenspunning] something between yacc and llvm ir, here.)

Anyway, I'm going to stop daydreaming about adding phi values before I talk
myself into rewriting this into a pile of SSA-generating macros-- as motivating
as I find that. There's a balance that needs to be struck between looking up at
where we're going and down at one's feet and --for now, at least-- it's time to
continue plodding forward.

[greenspunning]: https://en.wikipedia.org/wiki/Greenspun%27s_tenth_rule

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
