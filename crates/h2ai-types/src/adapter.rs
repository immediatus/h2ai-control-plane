use crate::config::AdapterKind;
use crate::physics::TauValue;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

/// Model provider family. Used for correlated hallucination detection at calibration time.
///
/// When all non-Mock adapters in the calibration pool belong to the same family,
/// their failures are correlated: the same hallucination appears in 2/3 of proposals
/// simultaneously, breaking the Weiszfeld BFT breakdown-point guarantee.
///
/// `Mock` is family-neutral: Mock adapters do not participate in family diversity checks.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdapterFamily {
    Anthropic,
    OpenAI,
    Google,
    Meta,
    Mistral,
    /// Local or self-hosted model (Ollama, llama.cpp, CloudGeneric endpoint, NATS dispatch).
    Local,
    /// Test double — exempt from multi-family enforcement.
    Mock,
}

impl std::fmt::Display for AdapterFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            AdapterFamily::Anthropic => "Anthropic",
            AdapterFamily::OpenAI => "OpenAI",
            AdapterFamily::Google => "Google",
            AdapterFamily::Meta => "Meta",
            AdapterFamily::Mistral => "Mistral",
            AdapterFamily::Local => "Local",
            AdapterFamily::Mock => "Mock",
        };
        write!(f, "{s}")
    }
}

impl From<&AdapterKind> for AdapterFamily {
    fn from(kind: &AdapterKind) -> Self {
        match kind {
            AdapterKind::Anthropic { .. } => AdapterFamily::Anthropic,
            AdapterKind::OpenAI { .. } => AdapterFamily::OpenAI,
            AdapterKind::Ollama { .. }
            | AdapterKind::CloudGeneric { .. }
            | AdapterKind::LocalLlamaCpp { .. } => AdapterFamily::Local,
        }
    }
}

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
}

#[async_trait]
pub trait IComputeAdapter: Send + Sync + std::fmt::Debug {
    async fn execute(&self, request: ComputeRequest) -> Result<ComputeResponse, AdapterError>;
    fn kind(&self) -> &AdapterKind;

    /// Provider family for correlated hallucination detection.
    /// Derived from `kind()` by default; override only for adapters whose `AdapterKind`
    /// does not unambiguously identify the family (e.g. `MockAdapter`).
    fn family(&self) -> AdapterFamily {
        AdapterFamily::from(self.kind())
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AdapterKind;
    use std::collections::HashSet;

    fn family_of(kind: AdapterKind) -> AdapterFamily {
        AdapterFamily::from(&kind)
    }

    #[test]
    fn adapter_kind_to_family_mapping() {
        assert_eq!(
            family_of(AdapterKind::Anthropic {
                api_key_env: "K".into(),
                model: "m".into()
            }),
            AdapterFamily::Anthropic
        );
        assert_eq!(
            family_of(AdapterKind::OpenAI {
                api_key_env: "K".into(),
                model: "m".into()
            }),
            AdapterFamily::OpenAI
        );
        assert_eq!(
            family_of(AdapterKind::Ollama {
                endpoint: "http://localhost".into(),
                model: "m".into()
            }),
            AdapterFamily::Local
        );
        assert_eq!(
            family_of(AdapterKind::CloudGeneric {
                endpoint: "http://x".into(),
                api_key_env: "K".into()
            }),
            AdapterFamily::Local
        );
        assert_eq!(
            family_of(AdapterKind::LocalLlamaCpp {
                model_path: "/m".into(),
                n_threads: 4
            }),
            AdapterFamily::Local
        );
    }

    #[test]
    fn single_family_detection_all_anthropic() {
        let families: HashSet<AdapterFamily> = [
            AdapterFamily::Anthropic,
            AdapterFamily::Anthropic,
            AdapterFamily::Anthropic,
        ]
        .into_iter()
        .collect();
        let non_mock: Vec<_> = families
            .iter()
            .filter(|f| **f != AdapterFamily::Mock)
            .collect();
        assert_eq!(
            non_mock.len(),
            1,
            "three identical families must collapse to one"
        );
    }

    #[test]
    fn single_family_detection_mixed_families() {
        let families: HashSet<AdapterFamily> = [AdapterFamily::Anthropic, AdapterFamily::OpenAI]
            .into_iter()
            .collect();
        let non_mock: Vec<_> = families
            .iter()
            .filter(|f| **f != AdapterFamily::Mock)
            .collect();
        assert_eq!(
            non_mock.len(),
            2,
            "two distinct families must not trigger single-family warning"
        );
    }

    #[test]
    fn mock_only_pool_exempt_from_enforcement() {
        let families: HashSet<AdapterFamily> = [AdapterFamily::Mock, AdapterFamily::Mock]
            .into_iter()
            .collect();
        let non_mock: Vec<_> = families
            .iter()
            .filter(|f| **f != AdapterFamily::Mock)
            .collect();
        assert!(
            non_mock.is_empty(),
            "all-Mock pool must be exempt (non_mock.len() == 0)"
        );
    }

    #[test]
    fn mixed_mock_and_real_counts_real_only() {
        let families: HashSet<AdapterFamily> = [
            AdapterFamily::Mock,
            AdapterFamily::Anthropic,
            AdapterFamily::Anthropic,
        ]
        .into_iter()
        .collect();
        let non_mock: Vec<_> = families
            .iter()
            .filter(|f| **f != AdapterFamily::Mock)
            .collect();
        assert_eq!(
            non_mock.len(),
            1,
            "Mock must be excluded; one real family remains"
        );
    }

    #[test]
    fn adapter_family_display() {
        assert_eq!(AdapterFamily::Anthropic.to_string(), "Anthropic");
        assert_eq!(AdapterFamily::OpenAI.to_string(), "OpenAI");
        assert_eq!(AdapterFamily::Mock.to_string(), "Mock");
        assert_eq!(AdapterFamily::Local.to_string(), "Local");
    }
}
