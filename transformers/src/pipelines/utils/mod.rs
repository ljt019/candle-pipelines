use super::cache::ModelOptions;
use crate::error::DeviceError;
use crate::Result;
use candle_core::{CudaDevice, Device};

pub mod builder;
pub use builder::{BasePipelineBuilder, StandardPipelineBuilder};

/// Request for a specific device, used by pipeline builders.
#[derive(Clone, Default)]
pub enum DeviceRequest {
    /// Use CPU (default behavior, matches HuggingFace Python API).
    #[default]
    Cpu,
    /// Select a specific CUDA device by index. Errors if unavailable.
    Cuda(usize),
    /// Provide an already constructed device.
    Explicit(Device),
}

impl DeviceRequest {
    /// Resolve the request into an actual [`Device`].
    pub fn resolve(self) -> Result<Device> {
        match self {
            DeviceRequest::Cpu => Ok(Device::Cpu),
            DeviceRequest::Cuda(i) => {
                CudaDevice::new_with_stream(i)
                    .map(Device::Cuda)
                    .map_err(|e| {
                        DeviceError::CudaInitFailed {
                            index: i,
                            reason: e.to_string(),
                        }
                        .into()
                    })
            }
            DeviceRequest::Explicit(d) => Ok(d),
        }
    }
}

/// Trait providing convenience methods for pipeline builders to select a device.
pub trait DeviceSelectable: Sized {
    /// Returns a mutable reference to the builder's internal [`DeviceRequest`].
    fn device_request_mut(&mut self) -> &mut DeviceRequest;

    /// Run the pipeline on CPU (this is the default).
    fn cpu(mut self) -> Self {
        *self.device_request_mut() = DeviceRequest::Cpu;
        self
    }

    /// Select a specific CUDA device by index.
    ///
    /// Returns an error at build time if the CUDA device is unavailable.
    fn cuda_device(mut self, index: usize) -> Self {
        *self.device_request_mut() = DeviceRequest::Cuda(index);
        self
    }

    /// Provide an explicit [`Device`].
    fn device(mut self, device: Device) -> Self {
        *self.device_request_mut() = DeviceRequest::Explicit(device);
        self
    }
}

/// Utility to generate a cache key combining model options and device location.
pub fn build_cache_key<O: ModelOptions>(options: &O, device: &Device) -> String {
    format!("{}-{:?}", options.cache_key(), device.location())
}
