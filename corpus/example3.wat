(module
  (import "env" "add" (func $add (param i32 i32) (result i32)))
  (memory 1)

  (func (export "foo") (result i32)
    (call $add (i32.const 1) (i32.const 3))
  )
)
