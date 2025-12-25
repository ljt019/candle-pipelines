# candle_pipelines v0.0.1

<!-- CI / Workflow Badges -->
[<img alt="crates.io" src="https://img.shields.io/crates/v/candle_pipelines.svg?style=for-the-badge&color=fc8d62&logo=rust" height="19">](https://crates.io/crates/candle_pipelines)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-candle__pipelines-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs" height="19">](https://docs.rs/candle_pipelines)
![CI](https://github.com/ljt019/candle_pipelines/actions/workflows/ci.yml/badge.svg)

> [!warning]
> ***This crate is under active development. APIs may change as features are still being added, and things tweaked.***

Simple, intuitive pipelines for local LLM inference in Rust, powered by [Candle](https://github.com/huggingface/candle). API inspired by Python's [Transformers](https://huggingface.co/docs/transformers).

## Available Pipelines

***Note**: Currently, models are accessible through these pipelines only. Direct model interface coming eventually!*

### Text Generation Pipeline

Generate text for various applications, supports general completions, as well as function/tool calling, and streamed responses.

---

**Qwen3**  
*Optimized for tool calling and structured output*

```markdown
 Parameter Sizes:
├── 0.6B
├── 1.7B
├── 4B
├── 8B
├── 14B
└── 32B
```

[→ View on HuggingFace](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f)

---

**Gemma3**  
*Google's models for general language tasks*

```markdown
 Parameter Sizes:
├── 1B
├── 4B
├── 12B
└── 27B
```

[→ View on HuggingFace](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d)

### Analysis Pipelines

ModernBERT powers three specialized analysis tasks with shared architecture:

---

#### **Fill Mask Pipeline**
*Complete missing words in text*

```markdown
 Available Sizes:
├── Base
└── Large
```

[→ View on HuggingFace](https://huggingface.co/answerdotai/ModernBERT-base)

---

#### **Sentiment Analysis Pipeline**
*Analyze emotional tone in multiple languages*

```markdown
 Available Sizes:
├── Base
└── Large
```

[→ View on HuggingFace](https://huggingface.co/clapAI/modernBERT-base-multilingual-sentiment)

---

#### **Zero-shot Classification Pipeline**
*Classify text without training examples*

```markdown
 Available Sizes:
├── Base
└── Large
```

[→ View on HuggingFace](https://huggingface.co/MoritzLaurer/ModernBERT-base-zeroshot-v2.0)

---

***Technical Note**: All ModernBERT pipelines share the same backbone architecture, loading task-specific finetuned weights as needed.*

## Usage

At this point in development the only way to interact with the models is through the given pipelines, I plan to eventually provide a simple interface to work with the models directly.

Inference will be quite slow at the moment, this is mostly due to not using the CUDA feature when compiling candle. I will be working on integrating this smoothly in future updates for much faster inference.

### Text Generation

There are two basic ways to generate text:

1. By providing a simple prompt string.
2. By providing a list of messages for chat-like interactions.

#### Providing a single prompt

Use the `completion` method for straightforward text generation from a single prompt string.

```rust
use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3Size};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Create the pipeline
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .temperature(0.7)
        .top_k(40)
        .build()
        .await?;

    // 2. Generate a completion
    let completion = pipeline.completion("What is the meaning of life?").await?;
    println!("{}", completion);

    Ok(())
}
```

#### Providing a list of messages

For more conversational interactions, you can pass a list of messages to the `completion` method.

The `Message` struct represents a single message in a chat and has a `role` (system, user, or assistant) and `content`. You can create messages using:

- `Message::system(content: &str)`: For system prompts.
- `Message::user(content: &str)`: For user prompts.
- `Message::assistant(content: &str)`: For model responses.

```rust
use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3Size, Message};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Create the pipeline
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .temperature(0.7)
        .top_k(40)
        .build()
        .await?;

    // 2. Create the messages
    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("What is the meaning of life?"),
    ];

    // 3. Generate a completion
    let completion = pipeline.completion(&messages).await?;
    println!("{}", completion);

    Ok(())
}
```

#### Tool Calling

Using tools with models is also made extremely easy, you just define tools using the `tool` macro and make sure to register them with the pipeline and you are good to go.

Using the tools is as easy as calling `completion_with_tools` after having tools registered to the pipeline.

```rust
use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3Size, tool, tools};
use candle_pipelines::error::Result;

// 1. Define the tools
#[tool]
/// Get the weather for a given city in degrees celsius.
fn get_temperature(city: String) -> Result<String> {
    Ok(format!("The temperature is 20 degrees celsius in {}.", city))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 2. Create the pipeline
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .max_len(8192)
        .build()
        .await?;

    // 3. Register the tools
    pipeline.register_tools(tools![get_temperature]).await;

    // 4. Get a completion
    let completion = pipeline.completion_with_tools("What's the weather like in Tokyo?").await?;
    println!("{}", completion);

    Ok(())
}
```

Tools can also be asynchronous, allowing you to perform network or file I/O directly inside the handler:

```rust
use candle_pipelines::error::Result;
use candle_pipelines::text_generation::tool;

#[tool]
/// Echoes a message after waiting for a bit.
async fn delayed_echo(message: String) -> Result<String> {
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    Ok(message)
}
```

#### Streaming Completions

For both regular and tool-assisted generation there are streaming versions:

- `completion_stream`
- `completion_stream_with_tools`

Instead of returning the completion these methods return a stream you can iterate on to receive tokens individually as they are generated by the model instead of just receiving them all at once at the end.
The stream is wrapped in a `CompletionStream` helper with methods like `collect()`
to gather the full response or `take(n)` to grab the first few chunks. Both
helpers now return a `Result` to surface any errors that may occur during
streaming.

```rust
use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3Size};
use futures::StreamExt;
use std::io::Write;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Create the pipeline
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .max_len(1024)
        .build()
        .await?;

    // 2. Get a completion using stream method
    let mut stream = pipeline.completion_stream(
        "Explain the concept of Large Language Models in simple terms.",
    ).await?;

    // 3. Do something with tokens as they are generated
    while let Some(tok) = stream.next().await {
        print!("{}", tok);
        std::io::stdout().flush().unwrap();
    }

    Ok(())
}
```

#### XML Parsing for Structured Output

You can build pipelines with XML parsing capabilities to handle structured outputs from models. This is particularly useful for parsing tool calls, and reasoning traces.

```rust
use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3Size, TagParts};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Build a pipeline with XML parsing for specific tags
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .max_len(1024)
        .build_xml(&["think", "tool_result", "tool_call"])
        .await?;

    // 2. Generate completion - returns Vec<Event> instead of String
    let events = pipeline.completion("Explain your reasoning step by step.").await?;

    // 3. Process events based on tags
    for event in events {
        match event.tag() {
            Some("think") => match event.part() {
                TagParts::Start => println!("[THINKING]"),
                TagParts::Content => print!("{}", event.get_content()),
                TagParts::End => println!("[END THINKING]"),
            },
            None => {
                // Regular content outside tags
                if event.part() == TagParts::Content {
                    print!("{}", event.get_content());
                }
            }
            _ => {}
        }
    }

    Ok(())
}
```

The XML parser also works with streaming completions, emitting events as XML tags are encountered in the stream. This enables real-time processing of structured outputs without waiting for the full response.

### Fill Mask (ModernBERT)

```rust
use candle_pipelines::fill_mask::{FillMaskPipelineBuilder, ModernBertSize};

fn main() -> anyhow::Result<()> {
    // 1. Build the pipeline
    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base).build()?;

    // 2. Fill the mask
    let prediction = pipeline.predict("The capital of France is [MASK].")?;

    println!("{}: {:.2}", prediction.word, prediction.score);
    // Output: Paris: 0.98
    Ok(())
}
```

### Sentiment Analysis (ModernBERT Finetune)

```rust
use candle_pipelines::sentiment::{SentimentAnalysisPipelineBuilder, ModernBertSize};

fn main() -> anyhow::Result<()> {
    // 1. Build the pipeline
    let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base).build()?;

    // 2. Analyze sentiment
    let result = pipeline.predict("I love using Rust for my projects!")?;

    println!("Sentiment: {} (confidence: {:.2})", result.label, result.score);
    // Output: Sentiment: positive (confidence: 0.98)
    Ok(())
}
```

### Zero-Shot Classification (ModernBERT NLI Finetune)

Zero-shot classification offers two methods for different use cases:

#### Single-Label Classification (`classify`)

Use when you want to classify text into one of several **mutually exclusive** categories. Probabilities sum to 1.0.

```rust
use candle_pipelines::zero_shot::{ZeroShotClassificationPipelineBuilder, ModernBertSize};

fn main() -> anyhow::Result<()> {
    // 1. Build the pipeline
    let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;

    // 2. Single-label classification
    let text = "The Federal Reserve raised interest rates.";
    let candidate_labels = &["economics", "politics", "technology", "sports"];
    let results = pipeline.classify(text, candidate_labels)?;

    println!("Text: {}", text);
    for result in results {
        println!("- {}: {:.4}", result.label, result.score);
    }
    // Example output (probabilities sum to 1.0):
    // - economics: 0.8721
    // - politics: 0.1134
    // - technology: 0.0098
    // - sports: 0.0047
    
    Ok(())
}
```

#### Multi-Label Classification (`classify_multi_label`)

Use when labels can be **independent** and multiple labels could apply to the same text. Returns raw entailment probabilities.

```rust
use candle_pipelines::zero_shot::{ZeroShotClassificationPipelineBuilder, ModernBertSize};

fn main() -> anyhow::Result<()> {
    // 1. Build the pipeline
    let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;

    // 2. Multi-label classification
    let text = "I love reading books about machine learning and artificial intelligence.";
    let candidate_labels = &["technology", "education", "reading", "science"];
    let results = pipeline.classify_multi_label(text, candidate_labels)?;

    println!("Text: {}", text);
    for result in results {
        println!("- {}: {:.4}", result.label, result.score);
    }
    // Example output (independent probabilities):
    // - technology: 0.9234
    // - education: 0.8456
    // - reading: 0.9567
    // - science: 0.7821
    
    Ok(())
}
```

## Future Plans

- Add more model families and sizes
- Support additional pipelines (summarization, classification)
