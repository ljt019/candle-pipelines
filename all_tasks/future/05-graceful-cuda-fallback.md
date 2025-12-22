# Graceful CUDA Fallback

## Summary
When user requests CUDA but it's unavailable, fall back to CPU with a warning instead of erroring.

## Problem
Current behavior when CUDA requested but unavailable:

```rust
let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
    .cuda_device(0)  // User wants CUDA
    .build()
    .await?;         // ERROR if no CUDA!
```

This makes code non-portable. Users writing apps that should "use CUDA if available" have to handle this themselves.

## Solution
Add a `cuda_if_available()` builder method that tries CUDA, falls back to CPU:

```rust
impl<M: TextGenerationModel> DeviceSelectable for TextGenerationPipelineBuilder<M> {
    /// Use CUDA if available, otherwise fall back to CPU.
    ///
    /// Unlike `cuda_device()`, this won't error if CUDA is unavailable.
    /// A warning is logged when falling back to CPU.
    ///
    /// # Example
    /// ```rust
    /// let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
    ///     .cuda_if_available()  // Uses CUDA if present, CPU otherwise
    ///     .build()
    ///     .await?;
    /// ```
    fn cuda_if_available(mut self) -> Self {
        *self.device_request_mut() = DeviceRequest::CudaIfAvailable;
        self
    }
}
```

Add new variant to `DeviceRequest`:

```rust
pub enum DeviceRequest {
    Default,           // Try CUDA 0, fall back to CPU (current behavior)
    Cpu,               // Force CPU
    Cuda(usize),       // Force specific CUDA device (error if unavailable)
    CudaIfAvailable,   // Try CUDA 0, fall back to CPU with warning
    Explicit(Device),  // User-provided device
}
```

Wait - `Default` already does this! The issue is just that `cuda_device(0)` errors.

## Revised Solution
Actually, `DeviceRequest::Default` already tries CUDA and falls back:

```rust
DeviceRequest::Default => {
    match CudaDevice::new_with_stream(0) {
        Ok(cuda) => Ok(Device::Cuda(cuda)),
        Err(_) => Ok(Device::Cpu),  // Silent fallback
    }
}
```

Options:
1. **Document this behavior** - users should use default, not `cuda_device(0)`
2. **Add `prefer_cuda()`** - explicit "try CUDA, fallback OK" method
3. **Log warning on fallback** - so users know they're on CPU

### Recommended: Option 2 + 3
```rust
/// Prefer CUDA if available, fall back to CPU with a warning.
/// 
/// This is the recommended way to write portable code that benefits
/// from GPU acceleration when available.
fn prefer_cuda(mut self) -> Self {
    *self.device_request_mut() = DeviceRequest::PreferCuda;
    self
}
```

```rust
DeviceRequest::PreferCuda => {
    match CudaDevice::new_with_stream(0) {
        Ok(cuda) => Ok(Device::Cuda(cuda)),
        Err(e) => {
            tracing::warn!("CUDA unavailable ({}), falling back to CPU", e);
            Ok(Device::Cpu)
        }
    }
}
```

## Files to Modify
- `src/pipelines/utils/mod.rs` - add `PreferCuda` variant, `prefer_cuda()` method
- `src/pipelines/text_generation/builder.rs` - inherits via `DeviceSelectable`

## Notes
- Keep `cuda_device(n)` as strict "must have CUDA" for users who need it
- `prefer_cuda()` is the portable option
- Could also add `prefer_cuda_device(n)` for specific device preference

