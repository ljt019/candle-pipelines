use super::cache::ModelOptions;
use crate::{Result, TransformersError};
use candle_core::{CudaDevice, Device};

pub mod builder;
pub use builder::{BasePipelineBuilder, StandardPipelineBuilder};

/// Request for a specific device, used by pipeline builders.
#[derive(Clone, Default)]
pub enum DeviceRequest {
    /// Use CUDA if available, otherwise CPU (default behavior).
    #[default]
    Default,
    /// Force CPU even if CUDA is available.
    Cpu,
    /// Select a specific CUDA device by index.
    Cuda(usize),
    /// Provide an already constructed device.
    Explicit(Device),
}

impl DeviceRequest {
    /// Resolve the request into an actual [`Device`].
    pub fn resolve(self) -> Result<Device> {
        match self {
            DeviceRequest::Default => {
                // Try CUDA 0, fall back to CPU
                match CudaDevice::new_with_stream(0) {
                    Ok(cuda) => Ok(Device::Cuda(cuda)),
                    Err(_) => Ok(Device::Cpu),
                }
            }
            DeviceRequest::Cpu => Ok(Device::Cpu),
            DeviceRequest::Cuda(i) => CudaDevice::new_with_stream(i)
                .map(Device::Cuda)
                .map_err(|e| TransformersError::Device(e.to_string())),
            DeviceRequest::Explicit(d) => Ok(d),
        }
    }
}

/// Trait providing convenience methods for pipeline builders to select a device.
pub trait DeviceSelectable: Sized {
    /// Returns a mutable reference to the builder's internal [`DeviceRequest`].
    fn device_request_mut(&mut self) -> &mut DeviceRequest;

    /// Force the pipeline to run on CPU.
    fn cpu(mut self) -> Self {
        *self.device_request_mut() = DeviceRequest::Cpu;
        self
    }

    /// Select a specific CUDA device by index.
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
