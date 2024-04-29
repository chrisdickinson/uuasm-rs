use std::collections::HashMap;

use crate::{
    nodes::Module,
    rt::{machine::Machine, Value},
};

struct TestState {
    named_modules: HashMap<String, Module<'static>>,
    last_instance: Option<Machine<'static>>,
}

pub(crate) fn assert_return(
    machine: &mut Machine<'_>,
    field: &str,
    args: &[Value],
    expected: &[Value],
    location: &str,
) -> anyhow::Result<()> {
    let returned = machine
        .call(field, args)
        .map_err(|e| anyhow::anyhow!(r#"failed to call "{field}" ({location}); error="{e:?}""#))?;
    assert_eq!(
        returned.len(),
        expected.len(),
        r#"result mismatch ({location}); returned={returned:?}; expected={expected:?}"#
    );
    for (_idx, (result, expected)) in returned.iter().zip(expected.iter()).enumerate() {
        result.bit_eq(expected).map_err(|e| {
            anyhow::anyhow!(r#"result mismatch ({location}); error="{e:?}"; returned={returned:?}; expected={expected:?}"#)
        })?;
    }

    Ok(())
}
