(module
  (func (export "foo")
    (block (result i32)
      (i32.const 13)
      (br 0)
      (i32.const 13)
    )
    drop
  )
)
