use criterion::{criterion_group, criterion_main, Criterion};
use uuasm_codec::parse;
use uuasm_ir::DefaultIRGenerator;
use uuasm_rt::Imports;

const WASM: &[u8] = include_bytes!("../../../vendor/wasm-r3/benchmarks/factorial/factorial.wasm");

pub fn basic(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("basic");
    let mut imports = Imports::new();

    imports.link_module(
        "main",
        uuasm_codec::parse(DefaultIRGenerator::new(), WASM).unwrap(),
    );
    let mut machine = imports.clone().instantiate().unwrap();

    group.measurement_time(std::time::Duration::from_secs(10));
    group.throughput(criterion::Throughput::Bytes(WASM.len() as u64));
    group.bench_function("instantiate_and_run", |bench| {
        bench.iter(|| {
            let _ = machine.call("main", "_start", &[]);
        })
    });

    group.finish();
}

criterion_group!(benches, basic,);
criterion_main!(benches);
