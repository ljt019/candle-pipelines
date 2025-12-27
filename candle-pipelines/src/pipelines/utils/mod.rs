use super::cache::ModelOptions;
use crate::error::{PipelineError, Result};
use candle_core::backend::BackendDevice;
use candle_core::{CudaDevice, Device};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;

pub mod builder;
pub use builder::{BasePipelineBuilder, StandardPipelineBuilder};

#[derive(Clone, Default)]
pub enum DeviceRequest {
    #[default]
    Cpu,
    Cuda(usize),
}

impl DeviceRequest {
    pub fn resolve(self) -> Result<Device> {
        match self {
            DeviceRequest::Cpu => Ok(Device::Cpu),
            DeviceRequest::Cuda(i) => {
                // Cache one CudaDevice per GPU to avoid stream mismatches when
                // reusing cached models. Synchronize before returning to ensure
                // any pending operations from previous models are complete.
                static CUDA_DEVICE_CACHE: Lazy<Mutex<HashMap<usize, CudaDevice>>> =
                    Lazy::new(|| Mutex::new(HashMap::new()));

                let mut cache = CUDA_DEVICE_CACHE.lock().unwrap();
                if let Some(dev) = cache.get(&i) {
                    // Sync stream before reuse to flush any pending ops
                    dev.synchronize().map_err(|e| {
                        PipelineError::Device(format!("Failed to sync CUDA device {i}: {e}"))
                    })?;
                    return Ok(Device::Cuda(dev.clone()));
                }

                let dev = CudaDevice::new_with_stream(i).map_err(|e| {
                    PipelineError::Device(format!(
                        "Failed to init CUDA device {i}: {e}. Try CPU as fallback."
                    ))
                })?;
                cache.insert(i, dev.clone());
                Ok(Device::Cuda(dev))
            }
        }
    }
}

macro_rules! impl_device_methods {
    (direct: $builder:ident < $($gen:ident : $bound:path),* >) => {
        impl<$($gen: $bound),*> $builder<$($gen),*> {
            /// Use CPU for inference (default).
            pub fn cpu(mut self) -> Self {
                self.device_request = crate::pipelines::utils::DeviceRequest::Cpu;
                self
            }

            /// Use a specific CUDA GPU for inference.
            pub fn cuda(mut self, index: usize) -> Self {
                self.device_request = crate::pipelines::utils::DeviceRequest::Cuda(index);
                self
            }
        }
    };

    (delegated: $builder:ident < $($gen:ident : $bound:path),* >) => {
        impl<$($gen: $bound),*> $builder<$($gen),*> {
            /// Use CPU for inference (default).
            pub fn cpu(mut self) -> Self {
                *self.0.device_request_mut() = crate::pipelines::utils::DeviceRequest::Cpu;
                self
            }

            /// Use a specific CUDA GPU for inference.
            pub fn cuda(mut self, index: usize) -> Self {
                *self.0.device_request_mut() = crate::pipelines::utils::DeviceRequest::Cuda(index);
                self
            }
        }
    };
}

pub(crate) use impl_device_methods;

pub fn build_cache_key<O: ModelOptions>(options: &O, device: &Device) -> String {
    format!("{}-{:?}", options.cache_key(), device.location())
}
