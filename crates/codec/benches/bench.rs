use criterion::{black_box, criterion_group, criterion_main, Criterion};
use uuasm_codec::{old_parse, Decoder};
use uuasm_nodes::EmptyIRGenerator;
use wasmparser::FunctionBody;

const WASM: &[u8] = include_bytes!("../../../src/testsuite/func.0.wasm");

pub fn basic(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("basic");
    group.measurement_time(std::time::Duration::from_secs(10));
    group.throughput(criterion::Throughput::Bytes(WASM.len() as u64));
    group
        .bench_function("empty_parse", |bench| {
            bench.iter(|| {
                let mut parser = Decoder::<EmptyIRGenerator, ()>::new(
                    uuasm_codec::AnyParser::Module(uuasm_codec::ModuleParser::default()),
                    uuasm_nodes::EmptyIRGenerator::new(),
                );
                let _ = parser.write(WASM);
                parser.flush().unwrap();
            })
        })
        .bench_function("new_parse", |bench| {
            bench.iter(|| {
                let mut parser = Decoder::default();
                let _ = parser.write(WASM);
                let result = parser.flush().unwrap();
                black_box(result);
            })
        })
        .bench_function("old_parse", |bench| {
            bench.iter(|| {
                let result = old_parse(WASM).unwrap();
                black_box(result);
            })
        })
        .bench_function("wasmtime_wasmparser_parse", |bench| {
            bench.iter(|| {
                let parser = wasmparser::Parser::new(0);
                for payload in parser.parse_all(WASM) {
                    let payload = payload.unwrap();
                    match payload {
                        wasmparser::Payload::Version { .. } => {}
                        wasmparser::Payload::TypeSection(p) => {
                            for item in p {
                                let item = item.unwrap();
                                black_box(item);
                            }
                        }
                        wasmparser::Payload::ImportSection(p) => {
                            for item in p {
                                let item = item.unwrap();
                                black_box(item);
                            }
                        }
                        wasmparser::Payload::FunctionSection(p) => {
                            for item in p {
                                let item = item.unwrap();
                                black_box(item);
                            }
                        }
                        wasmparser::Payload::TableSection(p) => {
                            for item in p {
                                let item = item.unwrap();
                                black_box(item);
                            }
                        }
                        wasmparser::Payload::MemorySection(p) => {
                            for item in p {
                                let item = item.unwrap();
                                black_box(item);
                            }
                        }
                        wasmparser::Payload::GlobalSection(p) => {
                            for item in p {
                                let item = item.unwrap();
                                black_box(item);
                            }
                        }
                        wasmparser::Payload::ExportSection(p) => {
                            for item in p {
                                let item = item.unwrap();
                                black_box(item);
                            }
                        }
                        wasmparser::Payload::ElementSection(p) => {
                            for item in p {
                                let item = item.unwrap();
                                black_box(item);
                            }
                        }
                        wasmparser::Payload::DataSection(p) => {
                            for item in p {
                                let item = item.unwrap();
                                black_box(item);
                            }
                        }
                        wasmparser::Payload::CodeSectionEntry(p) => {
                            let locals = FunctionBody::new(p.get_binary_reader());
                            for item in locals.get_locals_reader().unwrap() {
                                let item = item.unwrap();
                                black_box(item);
                            }

                            for item in p.get_operators_reader().unwrap() {
                                let item = item.unwrap();
                                black_box(item);
                            }
                        }
                        wasmparser::Payload::ModuleSection { .. } => {
                            todo!();
                        }
                        _ => {}
                    }
                }
            })
        });

    group.finish();
}

criterion_group!(benches, basic,);
criterion_main!(benches);
