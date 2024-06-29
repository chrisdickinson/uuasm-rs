# a tiny uuebassembly interpreter

### 2024 June 25:

still working on the items from yesterday (still! omg!)

mid-breaking-up the crate into sub-crates there was a strange set of errors that suggested that
I had accidentally blown away my rust stdlib? but I think that was illusory. Fixing the improper
cross-crate `impl <struct>`'s fixed the problem.

other notes:

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

### 2024 June 24:

up next:

- [x] fix the Take<P>::advance function (it currently faults on a subtraction)
- [x] finish type section
- [x] break up the workspace into crates: nodes, parser, runtime
- [x] break the parser up into files
- [x] target sub-productions to make it easier to write tests
- [ ] look into injecting parser middleware
- [ ] ask ourselves: is there an easy way to support "read me a vec of this type"?



---



here's the plan:

- kind of big break

