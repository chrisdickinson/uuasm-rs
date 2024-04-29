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
use crate::rt::*;

#[test]
fn test_0() {
    let spectest = crate::parse::parse(include_bytes!("../../spectest.wasm")).expect("could not parse spec test");
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
                value = "f32::NAN"
            elif value == 'nan:arithmetic':
                value = "f32::NAN"
            else:
                value = f"unsafe {{ std::mem::transmute::<[u8; 4], f32>({value}u32.to_le_bytes()) }}"

            return f"Value::F32({value})"
        case "f64":
            if value == 'nan:canonical':
                value = "f64::NAN"
            elif value == 'nan:arithmetic':
                value = "f64::NAN"
            else:
                value = f"unsafe {{ std::mem::transmute::<[u8; 8], f64>({value}u64.to_le_bytes()) }}"

            return f"Value::F64({value})"
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

            if command.get("name", None):
                named_module_to_varname[command["name"]] = "module%d" % module_count

            extra_imports = "\n".join([f"""
                imports{module_count}.link_module("{key}", {registered_modules[key]}.clone());
            """ for key in registered_modules])

            output.append(f"""
                let module{module_count} = crate::parse::parse(
                    include_bytes!({escaped_filename})
                ).expect("failed to parse \\"{filename}\\" (\\"{source_filename}\\"; line {line})");

                let mut imports{module_count} = Imports::new();
                imports{module_count}.link_module("spectest", spectest.clone());
                {extra_imports}

                let mut instance{module_count} = imports{module_count}
                    .instantiate(module{module_count}.clone())
                    .expect("could not instantiate module{module_count} (\\"{source_filename}\\"; line {line})"); 
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

                    instance_name = f"instance{module_count}"
                    if "module" in command["action"]:
                            instance_name = named_module_to_varname[command["action"]["module"]].replace("module", "instance")

                    field = repr(field)[1:-1]
                    field = re.sub("\\\\u([a-fA-F0-9]{2,4})", "\\\\u{\\1}", field)
                    field = re.sub("\\\\U([a-fA-F0-9]+)", lambda m: re.sub("^0+", "", m.group(1)), field)
                    field = re.sub("\\\\'", "'", field)
                    field = re.sub("\"", "\\\\\"", field)

                    if re.search(r"\\x[89a-f]", field) is not None:
                        eprintln(f"skipping invocation \"{source_filename}\"; line {line}: field is invalid unicode...")
                        continue

                    output.append(f"""
                    {{
                        let result = {instance_name}.call("{field}", &[{args}]).expect(r#"failed to call {field} ("{source_filename}"; line {line}")"#);
                        let expected = &[{expected}];
                        assert_eq!(result.len(), expected.len());
                        for (result, expected) in result.iter().zip(expected.iter()) {{
                            result.bit_eq(expected).expect(r#"result mismatch {field} ("{source_filename}"; line {line}")"#);
                        }}
                    }}
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

                    instance_name = f"instance{module_count}"
                    if "module" in command["action"]:
                            instance_name = named_module_to_varname[command["action"]["module"]].replace("module", "instance")

                    field_esc = json.dumps(json.dumps(field))
                    if '\\u' in field_esc or '\\b' in field_esc:
                        ...

                    output.append(f"""
                        let field: String = serde_json::from_str({field_esc}).expect("could not decode");
                        let expectation = format!(r#"expected {"{}"} to fail but got success ("{source_filename}"; line {line}")"#, field.as_str());
                        let result = {instance_name}.call(field.as_str(), &[{args}]).err().expect(&expectation);
                        assert!(result.to_string().find("{text}").is_some());
                    """)

        case "assert_invalid":
            ...
        case "assert_malformed":
            ...
        case "assert_uninstantiable":
            ...
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

                    output.append(f"""
                        let field: String = serde_json::from_str({field_esc}).expect("could not decode");
                        let expectation = format!(r#"failed to call {"{}"} ("{source_filename}"; line {line}")"#, field.as_str());
                        instance{module_count}.call(field.as_str(), &[{args}]).expect(&expectation);
                    """)

        case "register":
            val = named_module_to_varname[command["name"]] if command.get("name", None) else f"module{module_count}"
            registered_modules[command["as"]] = val

output.append("""
}
""")

output = '\n'.join(output)
with open(pathlib.Path(sys.argv[1].replace("-", "_")).with_suffix(".rs"), "w") as f:
    f.write(output)

