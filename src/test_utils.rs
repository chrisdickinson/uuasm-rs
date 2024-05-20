#![allow(dead_code)]
use std::{borrow::Cow, collections::HashMap};

use crate::{
    intern_map::InternMap, nodes::{Import, Name}, parse::parse, rt::{machine::Machine, Imports, Value}
};

#[derive(Default)]
pub(crate) struct TestState {
    instances: Vec<Machine<'static>>,
    named_instances: HashMap<usize, usize>,
    intern_map: InternMap
}

pub(crate) fn declare_module(
    state: &mut TestState,
    bytes: &'static [u8],
    name: Option<&'static str>,
    location: &str,
) -> anyhow::Result<usize> {

    let module = parse(bytes)
        .map_err(|e| anyhow::anyhow!(r#"failed to parse module ({location}); error="{e:?}""#))?;

    let mut imports = Imports::new();

    for Import { r#mod: Name(module), .. } in module.import_section().unwrap_or_default() {
        let Some(name_idx) = state.intern_map.get(module) else {
            anyhow::bail!("no provider for {module}");
        };
        // imports.link_instance(module, &state.instances[state.named_instances[&name_idx]]);
    }

    let instance = imports
        .instantiate(module.clone())
        .map_err(|e| anyhow::anyhow!("could not instantiate module ({location})"))?; 

    let name = name.map(Cow::Borrowed).unwrap_or_else(|| {
        Cow::Owned(format!("@{}", state.named_instances.len()))
    });
    let name_idx = state.intern_map.insert(name.as_ref());
    let instance_idx = state.instances.len();
    state.instances.push(instance);
    state.named_instances.insert(name_idx, instance_idx);
    Ok(instance_idx)
}

pub(crate) fn register(
    state: &mut TestState,
    name: &'static str,
    instance_idx: usize,
) {
    let name_idx = state.intern_map.insert(name);
    state.named_instances.insert(name_idx, instance_idx);
}

pub(crate) fn assert_return(
    state: &mut TestState,
    instance_idx: usize,
    field: &str,
    args: &[Value],
    expected: &[Value],
    location: &str,
) -> anyhow::Result<()> {
    let Some(mut machine) = state.instances.get_mut(instance_idx) else {
        anyhow::bail!("no machine by that idx");
    };
    let returned = machine
        .call(field, args)
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

pub(crate) fn assert_fail(
    state: &mut TestState,
    instance_idx: usize,
    field: &str,
    args: &[Value],
    text: &str,
    location: &str,
) -> anyhow::Result<()> {
    let Some(mut machine) = state.instances.get_mut(instance_idx) else {
        anyhow::bail!("no machine by that idx");
    };

    match machine.call(field, args) {
        Ok(result) => {
            anyhow::bail!(r#"expected call of "{field}" to fail but got success; {result:?} ({location})"#)
        }
        Err(e) => {
            if !e.to_string().contains(text) {
                anyhow::bail!(r#"expected call of "{field}" to fail with "{text}" but got "{e:?}" ({location})"#)
            }
        }
    }
    Ok(())
}

pub(crate) fn invoke_call(
    state: &mut TestState,
    instance_idx: usize,
    field: &str,
    args: &[Value],
    location: &str,
) -> anyhow::Result<()> {
    let Some(mut machine) = state.instances.get_mut(instance_idx) else {
        anyhow::bail!("no machine by that idx");
    };

    machine
        .call(field, args)
        .map_err(|e| anyhow::anyhow!(r#"failed to call "{field}" ({location}); error="{e:?}""#))?;

    Ok(())
}
