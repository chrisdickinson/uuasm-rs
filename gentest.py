#!/usr/bin/env python3
import sys
import json
import pathlib
import struct
import math
import re
from functools import reduce
from decimal import Decimal

def eprintln(*args, **kwargs):
    kwargs['file'] = sys.stderr
    return print(*args, **kwargs)

with open(sys.argv[1], 'r') as f:
    data = f.read()

data = json.loads(data)
output = []
module_count = 0

output.append("""
#![allow(unused_mut)]
#![allow(unused)]
use crate::*;
use crate::test_utils::*;

#[test]
fn test_0() -> anyhow::Result<()> {
    let mut state = TestState::new(); 
""")

def convf32(f):
    if f == 'nan:canonical':
        return 0b01111111110000000000000000000000
    if f == 'nan:arithmetic':
        return 0b01111111110000000000000000000000
    if f.startswith("nan:"):
        return 0b01111111111110000000000000000000 | int(nan.split(":")[1], 16)
    return int.from_bytes(struct.pack("f", float(f)))

def convf64(f):
    if f == 'nan:canonical':
        return 0b0111111111111000000000000000000000000000000000000000000000000000
    if f == 'nan:arithmetic':
        return 0b0111111111111000000000000000000000000000000000000000000000000000
    if f.startswith("nan:"):
        return 0b0111111111111000000000000000000000000000000000000000000000000000 | int(nan.split(":")[1], 16)

    return int.from_bytes(struct.pack("d", Decimal(f)))

def to_value(desc):
    value = desc["value"]
    match desc["type"]:
        case "i32":
            return f"Value::I32({value}u32 as i32)"
        case "i64":
            return f"Value::I64({value}u64 as i64)"
        case "f32":
            if value == 'nan:canonical':
                return "Value::F32CanonicalNaN"
            elif value == 'nan:arithmetic':
                return "Value::F32ArithmeticNaN"
            else:
                return f"Value::F32(f32::from_bits({value}u32))"
        case "f64":
            if value == 'nan:canonical':
                return "Value::F64CanonicalNaN"
            elif value == 'nan:arithmetic':
                return "Value::F64ArithmeticNaN"
            else:
                return f"Value::F64(f64::from_bits({value}u64))"
        case "v128":
            match desc["lane_type"]:
                case 'i8':
                    value = reduce(lambda acc, v: acc | (int(v[1]) << (8 * v[0])), enumerate(value), 0)
                case 'i16':
                    value = reduce(lambda acc, v: acc | (int(v[1]) << (16 * v[0])), enumerate(value), 0)
                case 'i32':
                    value = reduce(lambda acc, v: acc | (int(v[1]) << (32 * v[0])), enumerate(value), 0)
                case 'i64':
                    value = reduce(lambda acc, v: acc | (int(v[1]) << (64 * v[0])), enumerate(value), 0)
                case 'f32':
                    value = reduce(lambda acc, v: acc | (convf32(v[1]) << (32 * v[0])), enumerate(value), 0)
                case 'f64':
                    value = reduce(lambda acc, v: acc | (convf64(v[1]) << (64 * v[0])), enumerate(value), 0)

            return f"Value::V128({value}u128 as i128)"
        case "externref":
            if value == 'null':
                return "Value::RefNull"
            return f"Value::RefExtern({value})"
        case "funcref":
            if value == 'null':
                return "Value::RefNull"
            return f"Value::RefFunc({value})"

source_filename = data["source_filename"]
named_module_to_varname = {}
registered_modules = {}
for command in data["commands"]:
    line = command["line"]

    match command["type"]:
        case "module":
            module_count += 1
            filename = command["filename"]
            escaped_filename = json.dumps(command["filename"])

            name = "None"
            if command.get("name", None):
                name = "Some(\"%s\")" % command.get("name")

            output.append(f"""
                declare_module(
                    &mut state,
                    include_bytes!({escaped_filename}),
                    {name},
                    r#""{source_filename}"; line {line}"#
                )?;
            """)

        case "assert_return":
            expected = command["expected"]

            match command["action"]["type"]:
                case "invoke":
                    field = command["action"]["field"]

                    args = command["action"]["args"]
                    expected = command["expected"]
                    args = ",".join(map(to_value, args))
                    expected = ",".join(map(to_value, expected))

                    field = repr(field)[1:-1]
                    field = re.sub("\\\\u([a-fA-F0-9]{2,4})", "\\\\u{\\1}", field)
                    field = re.sub("\\\\U([a-fA-F0-9]+)", lambda m: re.sub("^0+", "", m.group(1)), field)
                    field = re.sub("\\\\'", "'", field)
                    field = re.sub("\"", "\\\\\"", field)

                    if re.search(r"\\x[89a-f]", field) is not None:
                        eprintln(f"skipping invocation \"{source_filename}\"; line {line}: field is invalid unicode...")
                        continue

                    modname = "None"
                    if command["action"].get("module", None):
                        modname = 'Some(r#"%s"#)' % command["action"]["module"]

                    output.append(f"""
                        assert_return(
                            &mut state,
                            {modname},
                            "{field}",
                            &[{args}],
                            &[{expected}],
                            r#""{source_filename}"; line {line}"#
                        )?;
                    """)

                case "get":
                    ...

        case "assert_trap":
            expected = command["expected"]

            match command["action"]["type"]:
                case "invoke":
                    field = command["action"]["field"]

                    args = command["action"]["args"]
                    expected = command["expected"]
                    args = ",".join(map(to_value, args))
                    text = command["text"]

                    modname = "None"
                    if command["action"].get("module", None):
                        modname = 'Some(r#"%s"#)' % command["action"]["module"]

                    output.append(f"""
                        assert_fail(
                            &mut state,
                            {modname},
                            "{field}",
                            &[{args}],
                            r#"{text}"#,
                            r#""{source_filename}"; line {line}"#
                        )?;
                    """)

        case "assert_invalid":
            if command["module_type"] == "text":
                continue

            filename = command["filename"]
            escaped_filename = json.dumps(command["filename"])
            text = command["text"]

            output.append(f"""
                assert_invalid(
                    include_bytes!({escaped_filename}),
                    r#"{text}"#,
                    r#""{source_filename}"; line {line}"#
                )?;
            """)


        case "assert_malformed":
            if command["module_type"] == "text":
                continue

            filename = command["filename"]
            escaped_filename = json.dumps(command["filename"])
            text = command["text"]

            output.append(f"""
                assert_malformed(
                    include_bytes!({escaped_filename}),
                    r#"{text}"#,
                    r#""{source_filename}"; line {line}"#
                )?;
            """)

        case "assert_uninstantiable":
            filename = command["filename"]
            escaped_filename = json.dumps(command["filename"])
            text = command["text"]

            output.append(f"""
                assert_uninstantiable(
                    &mut state,
                    include_bytes!({escaped_filename}),
                    r#"{text}"#,
                    r#""{source_filename}"; line {line}"#
                )?;
            """)

        case "assert_exhaustion":
            ...

        case "assert_unlinkable":
            ...
        case "action":
            match command["action"]["type"]:
                case "invoke":
                    field = command["action"]["field"]

                    args = command["action"]["args"]
                    expected = command["expected"]
                    args = ",".join(map(to_value, args))
                    expected = ",".join(map(to_value, expected))

                    field_esc = json.dumps(json.dumps(field))
                    if '\\u' in field_esc or '\\b' in field_esc:
                        ...

                    modname = "None"
                    if command["action"].get("module", None):
                        modname = 'Some(r#"%s"#)' % command["action"]["module"]

                    output.append(f"""
                        let field: String = serde_json::from_str({field_esc}).expect("could not decode");
                        invoke_call(
                            &mut state,
                            {modname},
                            field.as_str(),
                            &[{args}],
                            r#""{source_filename}"; line {line}"#
                        )?;
                    """)

        case "register":
            name = command["as"]
            output.append(f"""
                register(&mut state, r#"{name}"#, r#""{source_filename}"; line {line}"#)?;
            """)

output.append("""
    Ok(())
}
""")

output = '\n'.join(output)
with open(pathlib.Path(sys.argv[1].replace("-", "_")).with_suffix(".rs"), "w") as f:
    f.write(output)

