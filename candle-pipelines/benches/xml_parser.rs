use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use candle_pipelines::text_generation::XmlParserBuilder;

fn bench_parse_simple(c: &mut Criterion) {
    let mut parser = XmlParserBuilder::new().register_tag("think").build();
    let input = "<think>Hello world</think>";

    c.bench_function("parse_simple", |b| {
        b.iter(|| parser.parse(black_box(input)))
    });
}

fn bench_parse_sizes(c: &mut Criterion) {
    let mut parser = XmlParserBuilder::new().register_tag("think").build();
    let mut group = c.benchmark_group("parse_content_size");

    for size in [10, 100, 1000, 10000] {
        let input = format!("<think>{}</think>", "x".repeat(size));
        group.bench_with_input(BenchmarkId::from_parameter(size), &input, |b, input| {
            b.iter(|| parser.parse(black_box(input)))
        });
    }
    group.finish();
}

fn bench_parse_many_tags(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse_registered_tags");

    for tag_count in [1, 5, 10, 20] {
        let mut builder = XmlParserBuilder::new();
        for i in 0..tag_count {
            builder = builder.register_tag(format!("tag{}", i));
        }
        let mut parser = builder.build();
        let input = "<tag0>content</tag0>";

        group.bench_with_input(
            BenchmarkId::from_parameter(tag_count),
            &input,
            |b, input| b.iter(|| parser.parse(black_box(input))),
        );
    }
    group.finish();
}

fn bench_streaming_tokens(c: &mut Criterion) {
    let parser = XmlParserBuilder::new().register_tag("think").build();

    c.bench_function("streaming_tokens", |b| {
        b.iter(|| {
            // Simulate LLM token output - small chunks
            let tokens = ["<", "think", ">", "Hello", " ", "world", "</", "think", ">"]
                .into_iter()
                .map(|s| Ok::<_, candle_pipelines::error::PipelineError>(s.to_string()));

            let events: Vec<_> = parser.parse_iter(black_box(tokens)).collect();
            black_box(events)
        })
    });
}

fn bench_worst_case_many_angles(c: &mut Criterion) {
    let mut parser = XmlParserBuilder::new().register_tag("think").build();

    // Many < that aren't tags - forces parser to check each one
    let input = "a < b < c < d < e < f <think>x</think>";

    c.bench_function("many_angles_not_tags", |b| {
        b.iter(|| parser.parse(black_box(input)))
    });
}

criterion_group!(
    benches,
    bench_parse_simple,
    bench_parse_sizes,
    bench_parse_many_tags,
    bench_streaming_tokens,
    bench_worst_case_many_angles,
);
criterion_main!(benches);
