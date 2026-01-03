use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use candle_pipelines::text_generation::{XmlParser, XmlTag};

/// Single tag for simple benchmarks.
#[derive(Debug, Clone, PartialEq)]
enum SingleTag {
    Think,
}

impl XmlTag for SingleTag {
    fn from_tag_str(s: &str) -> Option<Self> {
        match s.trim() {
            "think" => Some(Self::Think),
            _ => None,
        }
    }

    fn as_tag_str(&self) -> &'static str {
        match self {
            Self::Think => "think",
        }
    }
}

impl SingleTag {
    fn parser() -> XmlParser<Self> {
        XmlParser::new()
    }
}

/// Multiple tags for enum-based benchmarks.
#[derive(Debug, Clone, PartialEq)]
enum ManyTags {
    Tag0,
    Tag1,
    Tag2,
    Tag3,
    Tag4,
    Tag5,
    Tag6,
    Tag7,
    Tag8,
    Tag9,
}

impl XmlTag for ManyTags {
    fn from_tag_str(s: &str) -> Option<Self> {
        match s.trim() {
            "tag0" => Some(Self::Tag0),
            "tag1" => Some(Self::Tag1),
            "tag2" => Some(Self::Tag2),
            "tag3" => Some(Self::Tag3),
            "tag4" => Some(Self::Tag4),
            "tag5" => Some(Self::Tag5),
            "tag6" => Some(Self::Tag6),
            "tag7" => Some(Self::Tag7),
            "tag8" => Some(Self::Tag8),
            "tag9" => Some(Self::Tag9),
            _ => None,
        }
    }

    fn as_tag_str(&self) -> &'static str {
        match self {
            Self::Tag0 => "tag0",
            Self::Tag1 => "tag1",
            Self::Tag2 => "tag2",
            Self::Tag3 => "tag3",
            Self::Tag4 => "tag4",
            Self::Tag5 => "tag5",
            Self::Tag6 => "tag6",
            Self::Tag7 => "tag7",
            Self::Tag8 => "tag8",
            Self::Tag9 => "tag9",
        }
    }
}

impl ManyTags {
    fn parser() -> XmlParser<Self> {
        XmlParser::new()
    }
}

fn bench_parse_simple(c: &mut Criterion) {
    let mut parser = SingleTag::parser();
    let input = "<think>Hello world</think>";

    c.bench_function("parse_simple", |b| {
        b.iter(|| parser.parse(black_box(input)))
    });
}

fn bench_parse_sizes(c: &mut Criterion) {
    let mut parser = SingleTag::parser();
    let mut group = c.benchmark_group("parse_content_size");

    for size in [10, 100, 1000, 10000] {
        let input = format!("<think>{}</think>", "x".repeat(size));
        group.bench_with_input(BenchmarkId::from_parameter(size), &input, |b, input| {
            b.iter(|| parser.parse(black_box(input)))
        });
    }
    group.finish();
}

fn bench_parse_many_tags_enum(c: &mut Criterion) {
    // Test with a larger enum (10 variants)
    let mut parser = ManyTags::parser();
    let input = "<tag0>content</tag0>";

    c.bench_function("parse_many_tags_enum", |b| {
        b.iter(|| parser.parse(black_box(input)))
    });
}

fn bench_streaming_tokens(c: &mut Criterion) {
    let parser = SingleTag::parser();

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
    let mut parser = SingleTag::parser();

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
    bench_parse_many_tags_enum,
    bench_streaming_tokens,
    bench_worst_case_many_angles,
);
criterion_main!(benches);
