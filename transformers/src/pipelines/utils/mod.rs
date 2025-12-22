use super::cache::ModelOptions;
use crate::error::DeviceError;
use crate::Result;
use candle_core::{CudaDevice, Device};

pub mod builder;
pub use builder::{BasePipelineBuilder, StandardPipelineBuilder};

#[derive(Clone, Default)]
pub enum DeviceRequest {
    #[default]
    Cpu,
    Cuda(usize),
    Explicit(Device),
}

impl DeviceRequest {
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

pub trait DeviceSelectable: Sized {
    fn device_request_mut(&mut self) -> &mut DeviceRequest;

    fn cpu(mut self) -> Self {
        *self.device_request_mut() = DeviceRequest::Cpu;
        self
    }

    fn cuda_device(mut self, index: usize) -> Self {
        *self.device_request_mut() = DeviceRequest::Cuda(index);
        self
    }

    fn device(mut self, device: Device) -> Self {
        *self.device_request_mut() = DeviceRequest::Explicit(device);
        self
    }
}

pub fn build_cache_key<O: ModelOptions>(options: &O, device: &Device) -> String {
    format!("{}-{:?}", options.cache_key(), device.location())
}
