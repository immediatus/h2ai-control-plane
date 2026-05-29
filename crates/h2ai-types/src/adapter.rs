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
    /// Raw reasoning trace from two-phase thinking models (e.g. `DeepSeek` R1 with `content` +
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
/// Named adapters can be registered via [`with_named`] and looked up by [`get_by_name`].
#[derive(Clone)]
pub struct AdapterRegistry {
    reasoning: Arc<dyn IComputeAdapter>,
    scoring: Option<Arc<dyn IComputeAdapter>>,
    structural: Option<Arc<dyn IComputeAdapter>>,
    named: std::collections::HashMap<String, Arc<dyn IComputeAdapter>>,
}

impl AdapterRegistry {
    /// Create a registry with only a reasoning adapter. Scoring and structural
    /// profiles fall back to the reasoning adapter until explicitly configured.
    #[must_use]
    pub fn new(reasoning: Arc<dyn IComputeAdapter>) -> Self {
        Self {
            reasoning,
            scoring: None,
            structural: None,
            named: std::collections::HashMap::new(),
        }
    }

    /// Attach a dedicated adapter for `TaskProfile::Scoring` tasks.
    #[must_use]
    pub fn with_scoring(mut self, adapter: Arc<dyn IComputeAdapter>) -> Self {
        self.scoring = Some(adapter);
        self
    }

    /// Attach a dedicated adapter for `TaskProfile::Structural` tasks.
    #[must_use]
    pub fn with_structural(mut self, adapter: Arc<dyn IComputeAdapter>) -> Self {
        self.structural = Some(adapter);
        self
    }

    /// Register an adapter under an arbitrary string name.
    ///
    /// Used for config-driven lookups such as `complexity_probe_adapter = "researcher"`.
    #[must_use]
    pub fn with_named(
        mut self,
        name: impl Into<String>,
        adapter: Arc<dyn IComputeAdapter>,
    ) -> Self {
        self.named.insert(name.into(), adapter);
        self
    }

    /// Look up an adapter by its registered name.
    ///
    /// Returns `Some` if the name was registered via [`with_named`], `None` otherwise.
    #[must_use]
    pub fn get_by_name(&self, name: &str) -> Option<&dyn IComputeAdapter> {
        self.named.get(name).map(Arc::as_ref)
    }

    /// Resolve the adapter for the given profile.
    ///
    /// `Scoring` and `Structural` fall back to the reasoning adapter when no
    /// dedicated adapter has been configured.
    #[must_use]
    pub fn resolve(&self, profile: &TaskProfile) -> &dyn IComputeAdapter {
        match profile {
            TaskProfile::Reasoning => self.reasoning.as_ref(),
            TaskProfile::Scoring => self
                .scoring
                .as_deref()
                .unwrap_or_else(|| self.reasoning.as_ref()),
            TaskProfile::Structural => self
                .structural
                .as_deref()
                .unwrap_or_else(|| self.reasoning.as_ref()),
        }
    }
}

impl std::fmt::Debug for AdapterRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let named_keys: Vec<&str> = self.named.keys().map(String::as_str).collect();
        f.debug_struct("AdapterRegistry")
            .field("reasoning", &self.reasoning.kind())
            .field("scoring", &self.scoring.as_ref().map(|a| a.kind()))
            .field("structural", &self.structural.as_ref().map(|a| a.kind()))
            .field("named", &named_keys)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AdapterKind;

    mockall::mock! {
        #[derive(Debug)]
        pub StubAdapter {}
        #[async_trait::async_trait]
        impl IComputeAdapter for StubAdapter {
            async fn execute(&self, req: ComputeRequest) -> Result<ComputeResponse, AdapterError>;
            fn kind(&self) -> &AdapterKind;
        }
    }

    fn stub_kind() -> AdapterKind {
        AdapterKind::CloudGeneric {
            endpoint: "stub://localhost".into(),
            api_key_env: "STUB".into(),
            model: None,
            provider: Default::default(),
        }
    }

    fn stub_adapter() -> MockStubAdapter {
        let mut m = MockStubAdapter::new();
        let kind = stub_kind();
        m.expect_execute().returning(move |_| {
            Ok(ComputeResponse {
                output: "stub".into(),
                token_cost: 0,
                adapter_kind: stub_kind(),
                tokens_used: None,
                reasoning_trace: None,
            })
        });
        m.expect_kind().return_const(kind).times(0..);
        m
    }

    #[test]
    fn adapter_registry_get_by_name_returns_named_adapter() {
        let reasoning = Arc::new(stub_adapter());
        let named_adapter = Arc::new(stub_adapter());

        let registry = AdapterRegistry::new(reasoning).with_named("my-adapter", named_adapter);

        assert!(
            registry.get_by_name("my-adapter").is_some(),
            "expected Some for registered name"
        );
        assert!(
            registry.get_by_name("nonexistent").is_none(),
            "expected None for unregistered name"
        );
    }
}
