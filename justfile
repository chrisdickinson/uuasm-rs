_help:
  @just --list

set positional-arguments
generate_test for_json interp="{{}}":

setup_tests:
  #!/bin/bash
  git submodule update --init --recursive
  rm -rf crates/rt/src/testsuite
  mkdir -p crates/rt/src/testsuite

  for i in $(find vendor/testsuite -name '*.wast' | grep -v proposals); do
    filename=$(basename "$i")
    filename=${filename%.wast}

    wast2json -o "$(pwd)/crates/rt/src/testsuite/${filename}.json" "$i" &
  done
  wait

  find crates/rt/src/testsuite -name '*.json' | xargs -P0 -I{} python3 gentest.py {}
  for mod in $(find crates/rt/src/testsuite -name '*.rs' | grep -v simd); do
    mod=${mod%.rs}

    mod=$(basename "$mod")
    case "$mod" in
      "type"|"const"|"loop"|"if"|"return")
        mod="r#$mod"
      ;;
    esac

    echo "mod $mod;" >> crates/rt/src/testsuite/mod.rs
  done

test:
  #!/bin/bash
  for file in $(find corpus -name '*.wat'); do
    wasm-tools parse $file | wasm-tools strip --all -o ${file%.wat}.wasm
  done
  cargo test

lint:
  cargo fmt --check

fmt:
  cargo fmt
  cargo clippy --fix --allow-dirty
