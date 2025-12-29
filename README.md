# candle-pipelines

<!-- CI / Workflow Badges -->
[<img alt="crates.io" src="https://img.shields.io/crates/v/candle-pipelines.svg?style=for-the-badge&color=fc8d62&logo=rust" height="19">](https://crates.io/crates/candle-pipelines)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-candle--pipelines-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs" height="19">](https://docs.rs/candle-pipelines)
![CI](https://github.com/ljt019/candle_pipelines/actions/workflows/ci.yml/badge.svg)

> [!warning]
> ***This crate is under active development. APIs may change as features are still being added, and things tweaked.***

Simple, intuitive pipelines for local LLM inference in Rust, powered by [Candle](https://github.com/huggingface/candle). API inspired by Python's [Transformers](https://huggingface.co/docs/transformers).

## Available Pipelines

***Note**: Currently, models are accessible through these pipelines only. Direct model interface coming eventually!*

### Text Generation Pipeline

Generate text for various applications. Supports completions, tool calling, and token-by-token iteration.

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

Use the `run` method for straightforward text generation from a single prompt string.

```rust
use candle_pipelines::error::Result;
use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3Size};

fn main() -> Result<()> {
    // 1. Create the pipeline
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .temperature(0.7)
        .top_k(40)
        .build()?;

    // 2. Generate a completion - returns Output { text, stats }
    let output = pipeline.run("What is the meaning of life?")?;
    println!("{}", output.text);
    println!("Generated {} tokens", output.stats.tokens_generated);

    Ok(())
}
```

#### Providing a list of messages

For more conversational interactions, you can pass a list of messages to the `run` method.

The `Message` struct represents a single message in a chat and has a `role` (system, user, assistant, or tool) and `content`. You can create messages using:

- `Message::system(content: &str)`: For system prompts.
- `Message::user(content: &str)`: For user prompts.
- `Message::assistant(content: &str)`: For model responses.
- `Message::tool(content: &str)`: For tool/function results returned to the model.

```rust
use candle_pipelines::error::Result;
use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3Size, Message};

fn main() -> Result<()> {
    // 1. Create the pipeline
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .temperature(0.7)
        .top_k(40)
        .build()?;

    // 2. Create the messages
    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("What is the meaning of life?"),
    ];

    // 3. Generate a completion
    let output = pipeline.run(&messages)?;
    println!("{}", output.text);

    Ok(())
}
```

#### Tool Calling

Using tools with models is also made extremely easy, you just define tools using the `#[tool]` macro, register them with the pipeline, and they're executed automatically when the model calls them.

```rust
use candle_pipelines::error::Result;
use candle_pipelines::text_generation::{tool, tools, ErrorStrategy};
use candle_pipelines::text_generation::{Qwen3Size, TextGenerationPipelineBuilder};

// 1. Define tools using the #[tool] macro
#[tool(retries = 5)]  // optional: configure retry attempts
/// Get the humidity for a given city.
fn get_humidity(city: String) -> Result<String> {
    Ok(format!("The humidity is 50% in {}.", city))
}

#[tool]  // defaults to 3 retries
/// Get the temperature for a given city in degrees celsius.
fn get_temperature(city: String) -> Result<String> {
    Ok(format!("The temperature is 20 degrees celsius in {}.", city))
}

fn main() -> Result<()> {
    // 2. Create the pipeline
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .max_len(8192)
        .tool_error_strategy(ErrorStrategy::ReturnToModel)  // let model handle tool errors
        .build()?;

    // 3. Register tools (enabled by default)
    pipeline.register_tools(tools![get_temperature, get_humidity]);

    // 4. Get a completion - tools are used automatically
    let output = pipeline.run("What's the temp and humidity like in Tokyo?")?;
    println!("{}", output.text);

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

#### Token Iteration

Use `run_iter` to receive tokens as they're generated. Fully sync - no async runtime needed.

```rust
use candle_pipelines::error::Result;
use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3Size};
use std::io::Write;

fn main() -> Result<()> {
    // 1. Create the pipeline
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .max_len(1024)
        .build()?;

    // 2. Iterate over tokens as they're generated
    let mut tokens = pipeline.run_iter(
        "Explain the concept of Large Language Models in simple terms.",
    )?;

    // 3. Print tokens as they arrive
    for tok in &mut tokens {
        print!("{}", tok?);
        std::io::stdout().flush().unwrap();
    }

    // 4. Get stats after iteration
    let stats = tokens.stats();
    println!("\n\nGenerated {} tokens", stats.tokens_generated);

    Ok(())
}
```

#### XML Parsing for Structured Output

Use `XmlParserBuilder` to parse structured outputs from models - useful for tool calls and reasoning traces.

```rust
use candle_pipelines::error::Result;
use candle_pipelines::text_generation::{
    Qwen3Size, TagParts, TextGenerationPipelineBuilder, XmlParserBuilder,
};

fn main() -> Result<()> {
    // 1. Build a regular pipeline
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .max_len(1024)
        .build()?;

    // 2. Create XML parser for specific tags
    let parser = XmlParserBuilder::new()
        .register_tag("think")
        .register_tag("tool_result")
        .register_tag("tool_call")
        .build();

    // 3. Get token iterator
    let tokens = pipeline.run_iter("Explain your reasoning step by step.")?;

    // 4. Wrap with XML parser
    let events = parser.parse(tokens);

    // 5. Process events
    for event in events {
        match event.tag() {
            Some("think") => match event.part() {
                TagParts::Start => println!("[THINKING]"),
                TagParts::Content => print!("{}", event.get_content()),
                TagParts::End => println!("[END THINKING]"),
            },
            None => match event.part() {
                TagParts::Content => print!("{}", event.get_content()),
                _ => {}
            },
            _ => {}
        }
    }

    Ok(())
}
```

The XML parser emits events as tags are encountered, enabling real-time processing without waiting for the full response.

### Fill Mask (ModernBERT)

```rust
use candle_pipelines::error::Result;
use candle_pipelines::fill_mask::{FillMaskPipelineBuilder, ModernBertSize};

fn main() -> Result<()> {
    // 1. Build the pipeline
    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base).build()?;

    // 2. Fill the mask
    let output = pipeline.run("The capital of France is [MASK].")?;

    println!("{}: {:.2}", output.prediction.token, output.prediction.score);
    // Output: Paris: 0.98
    Ok(())
}
```

### Sentiment Analysis (ModernBERT Finetune)

```rust
use candle_pipelines::error::Result;
use candle_pipelines::sentiment::{SentimentAnalysisPipelineBuilder, ModernBertSize};

fn main() -> Result<()> {
    // 1. Build the pipeline
    let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base).build()?;

    // 2. Analyze sentiment
    let output = pipeline.run("I love using Rust for my projects!")?;

    println!("Sentiment: {} (confidence: {:.2})", output.prediction.label, output.prediction.score);
    // Output: Sentiment: positive (confidence: 0.98)
    Ok(())
}
```

### Zero-Shot Classification (ModernBERT NLI Finetune)

Zero-shot classification offers two methods for different use cases:

#### Single-Label Classification (`run`)

Use when you want to classify text into one of several **mutually exclusive** categories. Probabilities sum to 1.0.

```rust
use candle_pipelines::error::Result;
use candle_pipelines::zero_shot::{ZeroShotClassificationPipelineBuilder, ModernBertSize};

fn main() -> Result<()> {
    // 1. Build the pipeline
    let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;

    // 2. Single-label classification
    let text = "The Federal Reserve raised interest rates.";
    let labels = &["economics", "politics", "technology", "sports"];
    let output = pipeline.run(text, labels)?;

    println!("Text: {}", text);
    for p in &output.predictions {
        println!("- {}: {:.4}", p.label, p.score);
    }
    // Example output (probabilities sum to 1.0):
    // - economics: 0.8721
    // - politics: 0.1134
    // - technology: 0.0098
    // - sports: 0.0047
    
    Ok(())
}
```

#### Multi-Label Classification (`run_multi_label`)

Use when labels can be **independent** and multiple labels could apply to the same text. Returns raw entailment probabilities.

```rust
use candle_pipelines::error::Result;
use candle_pipelines::zero_shot::{ZeroShotClassificationPipelineBuilder, ModernBertSize};

fn main() -> Result<()> {
    // 1. Build the pipeline
    let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;

    // 2. Multi-label classification
    let text = "I love reading books about machine learning and artificial intelligence.";
    let labels = &["technology", "education", "reading", "science"];
    let output = pipeline.run_multi_label(text, labels)?;

    println!("Text: {}", text);
    for p in &output.predictions {
        println!("- {}: {:.4}", p.label, p.score);
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
