use crate::config::AdapterKind;
use crate::sizing::TauValue;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeRequest {
    pub system_context: String,
    pub task: String,
    pub tau: TauValue,
    pub max_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeResponse {
    pub output: String,
    pub token_cost: u64,
    pub adapter_kind: AdapterKind,
    /// Tokens consumed by this response, when reported by the adapter.
    /// Used for token-cost β₀ EMA (`beta_from_token_spans`).
    #[serde(default)]
    pub tokens_used: Option<u64>,
    /// Raw reasoning trace from two-phase thinking models (e.g. DeepSeek R1 with `content` +
    /// `reasoning_content`).  `None` for standard models and reasoning-only models where the
    /// trace is promoted directly into `output`.  Preserved for Auditor Gate inspection.
    #[serde(default)]
    pub reasoning_trace: Option<String>,
}

#[derive(Debug, Error)]
pub enum AdapterError {
    #[error("adapter timed out before producing output")]
    Timeout,
    #[error("adapter OOM panic: {0}")]
    OomPanic(String),
    #[error("network error: {0}")]
    NetworkError(String),
    #[error("FFI error from llama.cpp: {0}")]
    FfiError(String),
    #[error("remote A2A agent returned failure: {0}")]
    Remote(String),
    #[error("remote A2A agent cancelled the task")]
    Cancelled,
    #[error("adapter unavailable — agent card fetch failed or task rejected")]
    Unavailable,
    #[error("adapter returned empty output after extraction pipeline")]
    EmptyOutput,
}

#[async_trait]
pub trait IComputeAdapter: Send + Sync + std::fmt::Debug {
    async fn execute(&self, request: ComputeRequest) -> Result<ComputeResponse, AdapterError>;
    fn kind(&self) -> &AdapterKind;
}

/// Capability tier required by a compute task.
///
/// Callsites declare which profile they need; `AdapterRegistry::resolve` returns
/// the configured adapter for that profile, falling back to `Reasoning` when a
/// dedicated adapter is not available.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskProfile {
    /// Full LLM — explorers, compound planning, any high-reasoning task.
    Reasoning,
    /// Small / cheap model — semantic similarity scoring, short JSON scoring tasks.
    /// Falls back to `Reasoning` if no dedicated adapter is configured.
    Scoring,
    /// Any model that reliably follows instructions — auditor, schema validation.
    /// Falls back to `Reasoning` if no dedicated adapter is configured.
    Structural,
}

/// Maps `TaskProfile` → `Arc<dyn IComputeAdapter>` with fallback to `Reasoning`.
///
/// Build with [`AdapterRegistry::new`] (requires only a reasoning adapter) and
/// optionally attach dedicated adapters via [`with_scoring`] / [`with_structural`].
#[derive(Clone)]
pub struct AdapterRegistry {
    reasoning: Arc<dyn IComputeAdapter>,
    scoring: Option<Arc<dyn IComputeAdapter>>,
    structural: Option<Arc<dyn IComputeAdapter>>,
}

impl AdapterRegistry {
    /// Create a registry with only a reasoning adapter. Scoring and structural
    /// profiles fall back to the reasoning adapter until explicitly configured.
    pub fn new(reasoning: Arc<dyn IComputeAdapter>) -> Self {
        Self {
            reasoning,
            scoring: None,
            structural: None,
        }
    }

    /// Attach a dedicated adapter for `TaskProfile::Scoring` tasks.
    pub fn with_scoring(mut self, adapter: Arc<dyn IComputeAdapter>) -> Self {
        self.scoring = Some(adapter);
        self
    }

    /// Attach a dedicated adapter for `TaskProfile::Structural` tasks.
    pub fn with_structural(mut self, adapter: Arc<dyn IComputeAdapter>) -> Self {
        self.structural = Some(adapter);
        self
    }

    /// Resolve the adapter for the given profile.
    ///
    /// `Scoring` and `Structural` fall back to the reasoning adapter when no
    /// dedicated adapter has been configured.
    pub fn resolve(&self, profile: &TaskProfile) -> &dyn IComputeAdapter {
        match profile {
            TaskProfile::Reasoning => self.reasoning.as_ref(),
            TaskProfile::Scoring => self.scoring.as_deref().unwrap_or(self.reasoning.as_ref()),
            TaskProfile::Structural => self
                .structural
                .as_deref()
                .unwrap_or(self.reasoning.as_ref()),
        }
    }
}

impl std::fmt::Debug for AdapterRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdapterRegistry")
            .field("reasoning", &self.reasoning.kind())
            .field("scoring", &self.scoring.as_ref().map(|a| a.kind()))
            .field("structural", &self.structural.as_ref().map(|a| a.kind()))
            .finish()
    }
}
