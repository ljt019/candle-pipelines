use super::cache::ModelOptions;
use crate::error::{Result, TransformersError};
use candle_core::{CudaDevice, Device};

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
                CudaDevice::new_with_stream(i)
                    .map(Device::Cuda)
                    .map_err(|e| {
                        TransformersError::Device(format!(
                            "Failed to init CUDA device {i}: {e}. Try CPU as fallback."
                        ))
                    })
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
