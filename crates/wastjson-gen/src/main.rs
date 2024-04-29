use std::{error::Error, fs::read_to_string};

use quote::quote;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
enum Value<'a> {
    F32(Option<&'a str>),
    I32(Option<&'a str>),
    F64(Option<&'a str>),
    I64(Option<&'a str>),
    V128(Option<Box<[&'a str]>>),
    #[serde(rename = "externref")]
    ExternRef(Option<&'a str>),
    #[serde(rename = "funcref")]
    FuncRef(Option<&'a str>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Action<'a> {
    Invoke {
        module: Option<&'a str>,
        field: &'a str,
        args: Box<[Value<'a>]>,
    },
    Get {
        module: Option<&'a str>,
        field: &'a str,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Input<'a> {
    source_filename: &'a str,
    commands: Box<[Command<'a>]>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound(deserialize = "'de: 'a"))]
struct Command<'a> {
    line: usize,
    #[serde(flatten)]
    kind: CommandKind<'a>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
enum CommandKind<'a> {
    Module {
        name: Option<&'a str>,
        filename: &'a str,
    },
    Register {
        name: Option<&'a str>,
        #[serde(rename = "as")]
        as_target: &'a str,
    },
    AssertReturn {
        action: Action<'a>,
        expected: Box<[Value<'a>]>,
    },
    AssertTrap {
        action: Action<'a>,
        text: &'a str,
        expected: Box<[Value<'a>]>,
    },
    AssertExhaustion {
        action: Action<'a>,
        text: &'a str,
        expected: Box<[Value<'a>]>,
    },
    AssertMalformed {
        text: &'a str,
        filename: &'a str,
    },
    AssertInvalid {
        text: &'a str,
        filename: &'a str,
    },
    AssertUnlinkable {
        text: &'a str,
        filename: &'a str,
    },
    AssertUninstantiable {
        text: &'a str,
        filename: &'a str,
    },
    Action {
        action: Action<'a>,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(arg) = std::env::args().nth(1) else {
        return Ok(());
    };

    let data = read_to_string(arg)?;
    let commands: Input = serde_json::from_str(data.as_str())?;
    println!("{commands:?}");

    let assert_return = quote! {
        fn assert_return(instance: &mut Machine<'a>, args: &[Value], expect: &[Value], line: usize) -> anyhow::Result<()> {
        }

    };

    Ok(())
}
