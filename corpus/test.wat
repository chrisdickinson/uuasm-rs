(module
  (type (;0;) (func (param i32) (result i32)))
  (type (;1;) (func (param i32)))
  (import "foo" "bar" (func (;0;) (type 0)))
  (import "bar" "baz" (global (;0;) i32))
  (import "alpha" "beta" (memory (;0;) 1))
  (import "gamma" "delta" (table (;0;) 1 4 funcref))
  (func (;1;) (type 1) (param i32)
    i32.const 0
  )
  (table (;1;) 1 funcref)
  (table (;2;) 1 2 funcref)
  (memory (;1;) 1)
  (memory (;2;) 1 4)
  (export "hey" (func 1))
  (data (;0;) "hello world")
)
