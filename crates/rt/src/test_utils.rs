#![allow(dead_code)]

use std::sync::Arc;

use crate::{imports::Imports, machine::Machine, value::Value};
use uuasm_codec::parse;
use uuasm_nodes::DefaultIRGenerator;

pub(crate) struct TestState {
    machine: Machine,
    names: Vec<Arc<str>>,
    decl_count: usize,
}

impl TestState {
    pub(crate) fn new() -> Self {
        let mut imports = Imports::new();
        imports.link_module(
            "spectest",
            parse(
                DefaultIRGenerator::new(),
                include_bytes!("../../../spectest.wasm"),
            )
            .expect("could not parse spectest"),
        );
        let machine = imports.instantiate().unwrap();
        Self {
            machine,
            names: vec!["spectest".into()],
            decl_count: 0,
        }
    }

    pub(crate) fn last_name(&self) -> &'static str {
        unsafe {
            let inp: &str = self.names[self.names.len() - 1].as_ref();
            let out: &'static str = core::mem::transmute(inp);
            out
        }
    }
}

pub(crate) fn declare_module(
    state: &mut TestState,
    bytes: &'static [u8],
    name: Option<&'static str>,
    location: &str,
) -> anyhow::Result<()> {
    let module = parse(DefaultIRGenerator::new(), bytes).map_err(|e| {
        anyhow::anyhow!(r#"failed to parse module ({location}); error="{e}" ({e:?})"#)
    })?;

    let name = name
        .map(str::to_string)
        .unwrap_or_else(|| format!("@instance{}", state.decl_count));
    state.names.push(name.as_str().into());
    state.decl_count += 1;

    state
        .machine
        .link_module(state.last_name(), module)
        .map_err(|e| anyhow::anyhow!("could not instantiate module ({location}); err={e:?}"))?;

    state
        .machine
        .init(state.last_name())
        .map_err(|e| anyhow::anyhow!("could not initialize module ({location}); err={e:?}"))?;
    Ok(())
}

pub(crate) fn register(
    state: &mut TestState,
    name: &'static str,
    location: &'static str,
) -> anyhow::Result<()> {
    state.machine.alias(state.last_name(), name)?;
    state.names.push(name.into());
    Ok(())
}

pub(crate) fn assert_return(
    state: &mut TestState,
    modname: Option<&str>,
    field: &str,
    args: &[Value],
    expected: &[Value],
    location: &str,
) -> anyhow::Result<()> {
    let returned = state
        .machine
        .call(modname.unwrap_or_else(|| state.last_name()), field, args)
        .map_err(|e| anyhow::anyhow!(r#"failed to call "{field}" ({location}); error="{e:?}""#))?;

    for (idx, (result, expected)) in returned.iter().zip(expected.iter()).enumerate() {
        result.bit_eq(expected).map_err(|e| {
            anyhow::anyhow!(r#"result mismatch at {idx} ({location}); error="{e:?}"; returned={returned:?}; expected={expected:?}"#)
        })?;
    }

    assert_eq!(
        returned.len(),
        expected.len(),
        r#"result mismatch ({location}); returned={returned:?}; expected={expected:?}"#
    );

    Ok(())
}

pub(crate) fn assert_uninstantiable(
    state: &mut TestState,
    bytes: &'static [u8],
    text: &str,
    location: &str,
) -> anyhow::Result<()> {
    let module = parse(DefaultIRGenerator::new(), bytes).map_err(|e| {
        anyhow::anyhow!(r#"failed to parse module ({location}); error="{e}" ({e:?})"#)
    })?;

    let name = format!("@instance{}", state.decl_count);
    state.names.push(name.as_str().into());
    state.decl_count += 1;

    state
        .machine
        .link_module(state.last_name(), module)
        .map_err(|e| anyhow::anyhow!("could not instantiate module ({location}); err={e:?}"))?;

    match state.machine.init(state.last_name()) {
        Ok(result) => {
            anyhow::bail!(
                r#"expected module instantiation to fail but got success; {result:?} ({location})"#
            )
        }
        Err(e) => {
            if !e.to_string().contains(text) {
                anyhow::bail!(
                    r#"expected module instantiation to fail with "{text}" but got "{e:?}" ({location})"#
                )
            }
        }
    }

    Ok(())
}

pub(crate) fn assert_invalid(
    bytes: &'static [u8],
    text: &str,
    location: &str,
) -> anyhow::Result<()> {
    match parse(DefaultIRGenerator::new(), bytes) {
        Ok(result) => {
            anyhow::bail!(
                r#"expected module validation to fail with "{text}" but got success; {result:?} ({location})"#
            )
        }
        Err(e) => {
            if !e.to_string().contains(text) {
                anyhow::bail!(
                    r#"expected module validation to fail with "{text}" but got "{e}" ({location})"#
                )
            }
        }
    }

    Ok(())
}

pub(crate) fn assert_malformed(
    bytes: &'static [u8],
    text: &str,
    location: &str,
) -> anyhow::Result<()> {
    match parse(DefaultIRGenerator::new(), bytes) {
        Ok(result) => {
            anyhow::bail!(
                r#"expected module instantiation to fail with "{text}" but got success; {result:?} ({location})"#
            )
        }
        Err(e) => {
            if !e.to_string().contains(text) {
                anyhow::bail!(
                    r#"expected module instantiation to fail with "{text}" but got "{e}" ({location})"#
                )
            }
        }
    }

    Ok(())
}

pub(crate) fn assert_fail(
    state: &mut TestState,
    modname: Option<&str>,
    field: &str,
    args: &[Value],
    text: &str,
    location: &str,
) -> anyhow::Result<()> {
    match state
        .machine
        .call(modname.unwrap_or_else(|| state.last_name()), field, args)
    {
        Ok(result) => {
            anyhow::bail!(
                r#"expected call of "{field}" to fail but got success; {result:?} ({location})"#
            )
        }
        Err(e) => {
            if !e.to_string().contains(text) {
                anyhow::bail!(
                    r#"expected call of "{field}" to fail with "{text}" but got "{e:?}" ({location})"#
                )
            }
        }
    }
    Ok(())
}

pub(crate) fn invoke_call(
    state: &mut TestState,
    modname: Option<&str>,
    field: &str,
    args: &[Value],
    location: &str,
) -> anyhow::Result<()> {
    state
        .machine
        .call(modname.unwrap_or_else(|| state.last_name()), field, args)
        .map_err(|e| anyhow::anyhow!(r#"failed to call "{field}" ({location}); error="{e:?}""#))?;

    Ok(())
}
