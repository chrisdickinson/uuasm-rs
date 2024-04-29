_help:
  @just --list

set positional-arguments
generate_test for_json interp="{{}}":

setup_tests:
  #!/bin/bash
  git submodule update --init --recursive
  rm -rf src/testsuite
  mkdir -p src/testsuite

  for i in $(find vendor/testsuite -name '*.wast' | grep -v proposals); do
    filename=$(basename "$i")
    filename=${filename%.wast}

    wast2json -o "$(pwd)/src/testsuite/${filename}.json" "$i" &
  done
  wait

  find src/testsuite -name '*.json' | xargs -P0 -I{} python3 gentest.py {}
  for mod in $(find src/testsuite -name '*.rs'); do
    if [ "$(wc -l $mod | awk '{print $1}')" -gt 5000 ]; then
      echo "skip $mod -- too long"
      continue
    fi

    mod=${mod%.rs}

    mod=$(basename "$mod")
    case "$mod" in
      "type"|"const"|"loop"|"if"|"return")
        mod="r#$mod"
      ;;
    esac

    echo "mod $mod;" >> src/testsuite/mod.rs
  done

test:
  #!/bin/bash
  wasm-tools parse example.wat | wasm-tools strip -a -o example.wasm
  cargo test

lint:
  cargo fmt --check

fmt:
  cargo fmt
  cargo clippy --fix --allow-dirty
