_help:
  @just --list

test:
  #!/bin/bash
  wasm-tools parse example.wat | wasm-tools strip -a -o example.wasm
  cargo test

lint:
  cargo fmt --check

fmt:
  cargo fmt
  cargo clippy --fix --allow-dirty
