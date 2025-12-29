use std::time::{Duration, Instant};

/// Statistics for pipeline execution.
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Total execution time.
    pub total_time: Duration,
    /// Number of items processed.
    pub items_processed: usize,
}

impl PipelineStats {
    /// Create a new stats tracker (call at start of operation).
    pub(crate) fn start() -> PipelineStatsBuilder {
        PipelineStatsBuilder {
            start_time: Instant::now(),
        }
    }
}

/// Builder for PipelineStats - tracks timing from creation to finalize.
pub(crate) struct PipelineStatsBuilder {
    start_time: Instant,
}

impl PipelineStatsBuilder {
    /// Finalize stats with the number of items processed.
    pub fn finish(self, items_processed: usize) -> PipelineStats {
        PipelineStats {
            total_time: self.start_time.elapsed(),
            items_processed,
        }
    }
}
