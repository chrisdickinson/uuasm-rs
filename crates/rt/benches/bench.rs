use criterion::{criterion_group, criterion_main, Criterion};
use uuasm_ir::DefaultIRGenerator;
use uuasm_rt::Imports;

const WASM: &[u8] = include_bytes!("../../../vendor/wasm-r3/benchmarks/jsc/jsc.wasm");

pub fn basic(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("basic");
    let mut imports = Imports::new();

    imports.link_module(
        "main",
        uuasm_codec::parse(DefaultIRGenerator::new(), WASM).unwrap(),
    );

    group.sample_size(10);
    group.bench_function("instantiate_and_run", |bench| {
        bench.iter_batched(
            || imports.clone().instantiate().unwrap(),
            |mut machine| {
                let _ = machine.call("main", "_start", &[]);
            },
            criterion::BatchSize::PerIteration,
        )
    });

    group.finish();
}

criterion_group!(benches, basic,);
criterion_main!(benches);
