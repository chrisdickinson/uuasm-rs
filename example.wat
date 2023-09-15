(module
  (func $add_i32 (export "add_i32") (param $p1 i32) (param $p2 i32) (result i32)
    ;; All Wasm items are accessed via their index.
    ;; $p1 and $p2 are just aliases for the numbers 0 and 1, respectively.
    ;; Therefore, we could substitute them below and it'd work fine, but this is clearer:
    get_local $p1
    get_local $p2
    i32.add
  )
)
