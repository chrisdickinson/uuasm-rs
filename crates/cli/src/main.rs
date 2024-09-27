use std::{env::args, error::Error};

use uuasm_ir::DefaultIRGenerator;
use uuasm_rt::Imports;

fn main() -> Result<(), Box<dyn Error>> {
    let Some(program) = args().nth(1) else {
        return Ok(());
    };
    eprintln!("{program}");

    let mut imports = Imports::new();

    let program_bytes = std::fs::read(program)?;

    imports.link_module(
        "main",
        uuasm_codec::parse(DefaultIRGenerator::new(), &program_bytes)?,
    );

    let mut machine = imports.instantiate()?;
    machine.call("main", "_start", &[])?;

    Ok(())
}
