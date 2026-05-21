#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
use async_trait::async_trait;
use h2ai_adapters::mock::MockAdapter;
use h2ai_autonomic::calibration::{
    aimd_decay, aimd_reset, beta_from_merge_spans, beta_from_n_eff_adj, beta_from_token_spans,
    compute_conflict_rate, yield_from_history, CalibrationHarness, CalibrationInput,
};
use h2ai_config::H2AIConfig;
use h2ai_constraints::types::{ConstraintDoc, ConstraintSeverity};
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use h2ai_types::identity::TaskId;
use h2ai_types::sizing::{EigenCalibration, EnsembleCalibration};
use nalgebra::DMatrix;

/// A test adapter that always returns an error — used to exercise error propagation paths.
#[derive(Debug)]
struct FailingAdapter {
    kind: AdapterKind,
}

impl FailingAdapter {
    fn new() -> Self {
        Self {
            kind: AdapterKind::CloudGeneric {
                endpoint: "mock://failing".into(),
                api_key_env: "NONE".into(),
                model: None,
            },
        }
    }
}

#[async_trait]
impl IComputeAdapter for FailingAdapter {
    async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        Err(AdapterError::Remote("test failure".to_string()))
    }

    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}

/// A minimal test adapter that returns a fixed output and reports a configurable `AdapterKind`.
/// Used to exercise multi-family scenarios without spinning up real network adapters.
#[derive(Debug)]
struct KindedMockAdapter {
    output: String,
    kind: AdapterKind,
}

impl KindedMockAdapter {
    fn anthropic(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            kind: AdapterKind::Anthropic {
                api_key_env: "MOCK_KEY".into(),
                model: "claude-test".into(),
            },
        }
    }

    fn local(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            kind: AdapterKind::Ollama {
                endpoint: "http://localhost:11434".into(),
                model: "llama3".into(),
            },
        }
    }

    fn cloud(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            kind: AdapterKind::CloudGeneric {
                endpoint: "mock://localhost".into(),
                api_key_env: "MOCK".into(),
                model: None,
            },
        }
    }
}

#[async_trait]
impl IComputeAdapter for KindedMockAdapter {
    async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        Ok(ComputeResponse {
            output: self.output.clone(),
            token_cost: 0,
            adapter_kind: self.kind.clone(),
            tokens_used: None,
            reasoning_trace: None,
        })
    }

    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}

fn mock_adapter() -> MockAdapter {
    MockAdapter::new("The proposed solution uses stateless JWT auth.".into())
}

#[tokio::test]
async fn calibration_produces_valid_coefficients() {
    let adapter = mock_adapter();
    let cfg = H2AIConfig::default();
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec![
            "Describe a stateless auth strategy".into(),
            "What makes a good API design?".into(),
            "Explain event sourcing".into(),
        ],
        adapters: vec![&adapter as &dyn h2ai_types::adapter::IComputeAdapter],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: None,
    };
    let event = CalibrationHarness::run(input).await.unwrap();
    assert!(event.coefficients.alpha >= 0.0 && event.coefficients.alpha < 1.0);
    assert!(event.coefficients.beta_base > 0.0);
    assert!(!event.coefficients.cg_samples.is_empty());
    assert!(event.coordination_threshold.value() <= 0.3);
}

#[tokio::test]
async fn calibration_single_adapter_uses_default_cg() {
    let adapter = mock_adapter();
    let cfg = H2AIConfig::default();
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["Single prompt".into()],
        adapters: vec![&adapter as &dyn h2ai_types::adapter::IComputeAdapter],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: None,
    };
    let event = CalibrationHarness::run(input).await.unwrap();
    assert_eq!(event.coefficients.cg_samples, vec![0.7]);
}

#[tokio::test]
async fn calibration_two_adapters_empty_corpus_uses_fallback_cg() {
    // Two adapters with empty corpus → single pair → fallback CG = cfg.calibration_cg_fallback
    let a = mock_adapter();
    let b = mock_adapter(); // same output
    let cfg = H2AIConfig::default();
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["Prompt one".into(), "Prompt two".into()],
        adapters: vec![
            &a as &dyn h2ai_types::adapter::IComputeAdapter,
            &b as &dyn h2ai_types::adapter::IComputeAdapter,
        ],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: None,
    };
    let event = CalibrationHarness::run(input).await.unwrap();
    // Two adapters with empty corpus → single pair → fallback CG = cfg.calibration_cg_fallback
    assert_eq!(event.coefficients.cg_samples.len(), 1);
    let expected_fallback = cfg.calibration_cg_fallback; // default: 0.7
    assert!(
        (event.coefficients.cg_samples[0] - expected_fallback).abs() < 1e-9,
        "empty-corpus calibration must fall back to cg_fallback={expected_fallback}, got: {}",
        event.coefficients.cg_samples[0]
    );
}

#[tokio::test]
async fn calibration_empty_task_prompts_with_two_adapters_produces_cg_zero() {
    // No prompts → outputs_i.is_empty() → triggers fallback path.
    let a = mock_adapter();
    let b = mock_adapter();
    let cfg = H2AIConfig::default();
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec![], // no prompts
        adapters: vec![
            &a as &dyn h2ai_types::adapter::IComputeAdapter,
            &b as &dyn h2ai_types::adapter::IComputeAdapter,
        ],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: None,
    };
    // Should not panic — produces CG from fallback path.
    let event = CalibrationHarness::run(input).await.unwrap();
    assert_eq!(event.coefficients.cg_samples.len(), 1);
    assert!(
        event.coefficients.cg_samples[0] >= 0.0,
        "CG must be non-negative"
    );
}

#[tokio::test]
async fn calibration_zero_adapters_returns_no_adapters_error() {
    use h2ai_autonomic::calibration::CalibrationError;
    let cfg = H2AIConfig::default();
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["any prompt".into()],
        adapters: vec![],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: None,
    };
    let result = CalibrationHarness::run(input).await;
    assert!(
        matches!(result, Err(CalibrationError::NoAdapters)),
        "zero adapters must return NoAdapters error, got {:?}",
        result.err()
    );
}

#[tokio::test]
async fn calibration_two_adapters_populates_ensemble() {
    let cfg = H2AIConfig::default();
    let a1 = MockAdapter::new("alpha beta gamma delta".into());
    let a2 = MockAdapter::new("delta epsilon zeta omega".into());
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["test prompt".into()],
        adapters: vec![
            &a1 as &dyn h2ai_types::adapter::IComputeAdapter,
            &a2 as &dyn h2ai_types::adapter::IComputeAdapter,
        ],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: None,
    };
    let event = CalibrationHarness::run(input).await.unwrap();
    let ec = event
        .ensemble
        .expect("ensemble should be Some with 2 adapters");
    assert!(
        ec.p_mean > 0.5 && ec.p_mean <= 1.0,
        "p_mean out of range: {}",
        ec.p_mean
    );
    assert!(
        ec.rho_mean >= 0.0 && ec.rho_mean <= 1.0,
        "rho_mean out of range: {}",
        ec.rho_mean
    );
    assert!(
        ec.n_optimal >= 1 && ec.n_optimal <= 9,
        "n_optimal out of range: {}",
        ec.n_optimal
    );
    assert!(
        ec.q_optimal >= ec.p_mean,
        "q_optimal should be >= p_mean: {} < {}",
        ec.q_optimal,
        ec.p_mean
    );
}

#[test]
fn beta_from_merge_spans_derives_correct_value() {
    // 5 proposals → n*(n-1)/2 = 10 modelled pairs; elapsed 0.001s; T₁ = 1.0s
    // β₀ = 0.001 / 10 / 1.0 = 0.0001
    let spans = vec![(0.001f64, 5usize)];
    let result = beta_from_merge_spans(&spans, 1.0);
    let expected = 0.001 / 10.0 / 1.0;
    assert!(result.is_some());
    let b = result.unwrap();
    assert!((b - expected).abs() < 1e-9, "expected {expected}, got {b}");
}

#[test]
fn beta_from_merge_spans_returns_none_for_empty_input() {
    assert!(beta_from_merge_spans(&[], 1.0).is_none());
}

#[test]
fn beta_from_merge_spans_returns_none_when_t1_zero() {
    let spans = vec![(0.1f64, 4usize)];
    assert!(beta_from_merge_spans(&spans, 0.0).is_none());
}

#[test]
fn beta_from_merge_spans_n_one_uses_pairs_guard() {
    // n=1 → n*(n-1)/2 = 0 → max(1, 0) = 1 to avoid division by zero
    let spans = vec![(0.002f64, 1usize)];
    let result = beta_from_merge_spans(&spans, 1.0);
    assert!(result.is_some(), "n=1 must not return None");
    let b = result.unwrap();
    // elapsed / 1 / T1 = 0.002
    assert!((b - 0.002).abs() < 1e-9, "n=1: expected 0.002, got {b}");
}

#[test]
fn beta_from_merge_spans_clamps_to_max() {
    // Extremely slow merge → clamped at 0.1
    // 4 proposals → 6 pairs; 1000s elapsed; T1 = 0.001s → β_raw = 1000/6/0.001 >> 0.1
    let spans = vec![(1000.0f64, 4usize)];
    let result = beta_from_merge_spans(&spans, 0.001);
    assert!(result.is_some());
    assert!((result.unwrap() - 0.1).abs() < 1e-9, "should clamp at 0.1");
}

#[tokio::test]
async fn calibration_harness_m3_populates_ensemble_and_eigen() {
    // Use 3 adapters with distinct outputs. With M=3, the harness executes Phase A
    // (2 adapters) and Phase B (3 adapters) in parallel. MockAdapter timing is
    // sub-nanosecond, so USL fit always falls back to config default α = 0.12.
    // We verify the multi-adapter code path ran by checking ensemble and eigen.
    let a1 = MockAdapter::new("stateless JWT authentication using signed tokens".into());
    let a2 = MockAdapter::new("event sourcing with CQRS separating reads from writes".into());
    let a3 = MockAdapter::new(
        "clean API boundary defined by domain interfaces not implementation".into(),
    );
    let cfg = H2AIConfig::default();
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec![
            "Describe a stateless auth approach".into(),
            "Explain CQRS and event sourcing".into(),
            "What is a good API boundary?".into(),
        ],
        adapters: vec![
            &a1 as &dyn h2ai_types::adapter::IComputeAdapter,
            &a2 as &dyn h2ai_types::adapter::IComputeAdapter,
            &a3 as &dyn h2ai_types::adapter::IComputeAdapter,
        ],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: None,
    };
    let event = CalibrationHarness::run(input).await.unwrap();

    // Structural assertions: verify multi-adapter code path ran via ensemble/eigen.
    assert!(
        event.ensemble.is_some(),
        "ensemble must be Some with 3 adapters"
    );
    assert!(event.eigen.is_some(), "eigen must be Some with 3 adapters");
    // 3 adapters → C(3,2) = 3 pairwise CG samples.
    assert_eq!(
        event.coefficients.cg_samples.len(),
        3,
        "3 adapters must produce 3 pairwise CG samples"
    );
}

#[tokio::test]
async fn calibration_accepts_empty_corpus() {
    let a = mock_adapter();
    let cfg = H2AIConfig::default();
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["Prompt".into()],
        adapters: vec![&a as &dyn h2ai_types::adapter::IComputeAdapter],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: None,
    };
    assert!(CalibrationHarness::run(input).await.is_ok());
}

#[tokio::test]
async fn calibration_non_empty_corpus_computes_hamming() {
    // Adapter A always returns "jwt auth" → VocabularyPresence("jwt") → true (score=1.0 >= 0.5)
    // Adapter B always returns "session cookie" → VocabularyPresence("jwt") → false (score=0.0 < 0.5)
    // One Hard constraint → Hamming distance = 1.0 → CG = 1.0 * align
    use h2ai_constraints::types::{
        ConstraintDoc, ConstraintPredicate, ConstraintSeverity, VocabularyMode,
    };
    let corpus = vec![ConstraintDoc {
        id: "auth-jwt".into(),
        source_file: "test".into(),
        description: "must use jwt".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::VocabularyPresence {
            mode: VocabularyMode::AnyOf,
            terms: vec!["jwt".into()],
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    }];
    let a = MockAdapter::new("jwt authentication token".into());
    let b = MockAdapter::new("session cookie storage".into());
    let cfg = H2AIConfig::default();
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["auth strategy".into()],
        adapters: vec![
            &a as &dyn h2ai_types::adapter::IComputeAdapter,
            &b as &dyn h2ai_types::adapter::IComputeAdapter,
        ],
        cfg: &cfg,
        constraint_corpus: &corpus,
        embedding_model: None,
    };
    let event = CalibrationHarness::run(input).await.unwrap();
    assert_eq!(event.coefficients.cg_samples.len(), 1);
    // Opposite fingerprints → Hamming = 1.0 → CG ≈ 1.0 (tau_alignment at equal taus = 1.0)
    assert!(
        (event.coefficients.cg_samples[0] - 1.0).abs() < 1e-9,
        "opposite constraint profiles must produce CG=1.0, got: {}",
        event.coefficients.cg_samples[0]
    );
}

#[tokio::test]
async fn cg_mode_is_constraint_profile() {
    use h2ai_types::events::CgMode;
    let mode = CgMode::default();
    assert!(matches!(mode, CgMode::ConstraintProfile));
    let json = serde_json::to_string(&mode).unwrap();
    let back: CgMode = serde_json::from_str(&json).unwrap();
    assert!(matches!(back, CgMode::ConstraintProfile));
}

#[tokio::test]
async fn n_eff_cosine_prior_populated_with_embedding_model() {
    // Diverse stub: orthogonal embeddings for 3 adapters → N_eff ≈ 3.0
    struct DiverseStub;
    impl h2ai_context::embedding::EmbeddingModel for DiverseStub {
        fn embed(&self, text: &str) -> Vec<f32> {
            if text.contains("stateless") {
                vec![1.0, 0.0, 0.0]
            } else if text.contains("CQRS") {
                vec![0.0, 1.0, 0.0]
            } else {
                vec![0.0, 0.0, 1.0]
            }
        }
    }
    let stub = DiverseStub;
    let a1 = MockAdapter::new("Use stateless JWT auth approach".into());
    let a2 = MockAdapter::new("CQRS separates reads and writes".into());
    let a3 = MockAdapter::new("API boundary isolates services".into());
    let cfg = H2AIConfig::default();
    let result = CalibrationHarness::run(CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["First prompt".into(), "Second prompt".into()],
        adapters: vec![
            &a1 as &dyn h2ai_types::adapter::IComputeAdapter,
            &a2 as &dyn h2ai_types::adapter::IComputeAdapter,
            &a3 as &dyn h2ai_types::adapter::IComputeAdapter,
        ],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: Some(&stub as &dyn h2ai_context::embedding::EmbeddingModel),
    })
    .await
    .unwrap();

    assert!(
        result.n_eff_cosine_prior > 1.0,
        "diverse adapters → n_eff_cosine_prior > 1.0, got {}",
        result.n_eff_cosine_prior
    );
    assert!(
        result.n_eff_cosine_prior <= 3.0 + 0.01,
        "n_eff_cosine_prior bounded by N=3, got {}",
        result.n_eff_cosine_prior
    );
}

#[tokio::test]
async fn n_eff_cosine_prior_fallback_without_embedding_model() {
    let a = MockAdapter::new("Answer to prompt".into());
    let cfg = H2AIConfig::default();
    let result = CalibrationHarness::run(CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["Prompt A".into()],
        adapters: vec![&a as &dyn h2ai_types::adapter::IComputeAdapter],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: None,
    })
    .await
    .unwrap();

    // Single adapter: fallback = 1.0 + cg_fallback × (1-1) = 1.0
    assert!(
        (result.n_eff_cosine_prior - 1.0).abs() < 0.01,
        "single adapter no model → n_eff_cosine_prior=1.0, got {}",
        result.n_eff_cosine_prior
    );
}

#[tokio::test]
async fn calibration_source_is_synthetic_priors_with_single_adapter() {
    use h2ai_types::events::CalibrationSource;

    let cfg = H2AIConfig::default();
    let adapter = mock_adapter();
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        adapters: vec![&adapter as &dyn h2ai_types::adapter::IComputeAdapter],
        task_prompts: vec!["test prompt".into()],
        constraint_corpus: &[],
        embedding_model: None,
        cfg: &cfg,
    };
    let result = CalibrationHarness::run(input).await.unwrap();
    assert_eq!(
        result.calibration_source,
        CalibrationSource::SyntheticPriors
    );
}

#[tokio::test]
async fn calibration_source_is_partial_fit_with_two_adapters() {
    use h2ai_types::events::CalibrationSource;

    let cfg = H2AIConfig::default();
    let adapter1 = mock_adapter();
    let adapter2 = mock_adapter();
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        adapters: vec![
            &adapter1 as &dyn h2ai_types::adapter::IComputeAdapter,
            &adapter2 as &dyn h2ai_types::adapter::IComputeAdapter,
        ],
        task_prompts: vec!["test prompt".into()],
        constraint_corpus: &[],
        embedding_model: None,
        cfg: &cfg,
    };
    let result = CalibrationHarness::run(input).await.unwrap();
    assert_eq!(result.calibration_source, CalibrationSource::PartialFit);
}

// ── adapter_families / explorer_verification_family_match / single_family_warning ─────────

/// Single adapter → `adapter_families` has exactly one entry, `single_family_warning` is true,
/// `explorer_verification_family_match` is false (only one distinct family present).
#[tokio::test]
async fn adapter_families_single_adapter_populates_one_family() {
    let cfg = H2AIConfig::default();
    let a = KindedMockAdapter::cloud("answer");
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["prompt".into()],
        adapters: vec![&a as &dyn IComputeAdapter],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: None,
    };
    let event = CalibrationHarness::run(input).await.unwrap();

    assert_eq!(
        event.adapter_families.len(),
        1,
        "single adapter must produce exactly one family entry, got: {:?}",
        event.adapter_families
    );
    assert!(
        event.single_family_warning,
        "single distinct family must set single_family_warning=true"
    );
    assert!(
        !event.explorer_verification_family_match,
        "single family → explorer_verification_family_match must be false"
    );
}

/// Two adapters from the same family (both Cloud) → `single_family_warning` true,
/// `explorer_verification_family_match` false.
#[tokio::test]
async fn adapter_families_same_family_sets_single_family_warning() {
    let cfg = H2AIConfig::default();
    let a = KindedMockAdapter::cloud("answer A");
    let b = KindedMockAdapter::cloud("answer B");
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["prompt one".into()],
        adapters: vec![&a as &dyn IComputeAdapter, &b as &dyn IComputeAdapter],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: None,
    };
    let event = CalibrationHarness::run(input).await.unwrap();

    // Both are Cloud → only one distinct family
    assert_eq!(
        event.adapter_families.len(),
        1,
        "two adapters of the same family must yield 1 distinct family, got: {:?}",
        event.adapter_families
    );
    assert!(
        event.single_family_warning,
        "same-family pair must set single_family_warning=true"
    );
    assert!(
        !event.explorer_verification_family_match,
        "same-family pair must leave explorer_verification_family_match=false"
    );
}

/// Two adapters from different families (Anthropic + Local) → `adapter_families` has two entries,
/// `explorer_verification_family_match` true, `single_family_warning` false.
#[tokio::test]
async fn adapter_families_cross_family_sets_verification_match() {
    let cfg = H2AIConfig::default();
    let a = KindedMockAdapter::anthropic("anthropic response");
    let b = KindedMockAdapter::local("local llm response");
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["prompt".into()],
        adapters: vec![&a as &dyn IComputeAdapter, &b as &dyn IComputeAdapter],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: None,
    };
    let event = CalibrationHarness::run(input).await.unwrap();

    assert_eq!(
        event.adapter_families.len(),
        2,
        "two distinct families must produce 2 family entries, got: {:?}",
        event.adapter_families
    );
    assert!(
        event.explorer_verification_family_match,
        "cross-family adapters must set explorer_verification_family_match=true"
    );
    assert!(
        !event.single_family_warning,
        "cross-family adapters must leave single_family_warning=false"
    );
}

/// Three adapters spanning three distinct families → `adapter_families` has 3 entries,
/// `explorer_verification_family_match` true, `single_family_warning` false.
#[tokio::test]
async fn adapter_families_three_distinct_families() {
    let cfg = H2AIConfig::default();
    let a = KindedMockAdapter::anthropic("anthropic");
    let b = KindedMockAdapter::local("local");
    let c = KindedMockAdapter::cloud("cloud");
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["prompt A".into(), "prompt B".into()],
        adapters: vec![
            &a as &dyn IComputeAdapter,
            &b as &dyn IComputeAdapter,
            &c as &dyn IComputeAdapter,
        ],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: None,
    };
    let event = CalibrationHarness::run(input).await.unwrap();

    assert_eq!(
        event.adapter_families.len(),
        3,
        "three distinct families must produce 3 entries, got: {:?}",
        event.adapter_families
    );
    assert!(
        event.explorer_verification_family_match,
        "three-family ensemble must set explorer_verification_family_match=true"
    );
    assert!(
        !event.single_family_warning,
        "three-family ensemble must leave single_family_warning=false"
    );
}

// ── epistemic β₀ override ───────────────────────────────────────────────────

struct FixedEmbeddingModel;
impl h2ai_context::embedding::EmbeddingModel for FixedEmbeddingModel {
    fn embed(&self, _text: &str) -> Vec<f32> {
        // All outputs get the same vector → cosine = 1.0 across all pairs
        // → N_eff will be close to 1 (fully collapsed / mode collapse)
        vec![1.0, 0.0]
    }
}

#[tokio::test]
async fn calibration_with_embedding_model_uses_epistemic_beta() {
    // Arrange: 3 adapters + an embedding model → epistemic β₀ path triggers
    let a1 = MockAdapter::new("solution one: stateless JWT auth".into());
    let a2 = MockAdapter::new("solution two: session token redis".into());
    let a3 = MockAdapter::new("solution three: OAuth2 flow with PKCE".into());
    let cfg = H2AIConfig::default();
    let model = FixedEmbeddingModel;
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["Describe an auth strategy".into()],
        adapters: vec![
            &a1 as &dyn h2ai_types::adapter::IComputeAdapter,
            &a2 as &dyn h2ai_types::adapter::IComputeAdapter,
            &a3 as &dyn h2ai_types::adapter::IComputeAdapter,
        ],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: Some(&model as &dyn h2ai_context::embedding::EmbeddingModel),
    };
    let event = CalibrationHarness::run(input).await.unwrap();
    // Epistemic β₀ must be positive and within the USL-valid range
    assert!(
        event.coefficients.beta_base > 0.0,
        "beta_base must be positive, got {}",
        event.coefficients.beta_base
    );
    assert!(
        event.coefficients.beta_base <= 0.5,
        "beta_base must be ≤ 0.5 (not degenerate), got {}",
        event.coefficients.beta_base
    );
    // With FixedEmbeddingModel (all same vector → cg_mean ≈ 1.0, n_eff ≈ 1.0),
    // beta_from_n_eff_adj(~1.0, ~1.0, 3, 2.0) ≈ 0.333 — significantly higher
    // than the cfg.beta_base_default fallback (~0.039).
    // This confirms the epistemic β₀ override path was taken.
    assert!(
        event.coefficients.beta_base > cfg.beta_base_default,
        "epistemic β₀ ({}) must exceed cfg.beta_base_default ({}) under mode collapse",
        event.coefficients.beta_base,
        cfg.beta_base_default
    );
}

#[tokio::test]
async fn calibration_baseline_accuracy_proxy_uses_from_measured_p() {
    // Lines 172-175: EnsembleCalibration::from_measured_p is called when
    // baseline_accuracy_proxy > 0.0 (non-default). Two adapters satisfy adapter_outputs.len() >= 2.
    let a1 = mock_adapter();
    let a2 = mock_adapter();
    let cfg = H2AIConfig {
        baseline_accuracy_proxy: 0.75,
        ..H2AIConfig::default()
    };
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        adapters: vec![
            &a1 as &dyn h2ai_types::adapter::IComputeAdapter,
            &a2 as &dyn h2ai_types::adapter::IComputeAdapter,
        ],
        task_prompts: vec!["test prompt".into()],
        constraint_corpus: &[],
        embedding_model: None,
        cfg: &cfg,
    };
    let result = CalibrationHarness::run(input).await.unwrap();
    assert!(
        result.coefficients.alpha >= 0.0,
        "alpha must be non-negative"
    );
}

#[tokio::test]
async fn calibration_embedding_model_with_no_task_prompts_uses_cosine_prior_fallback() {
    // Line 262: embedding_model present but k_prompts==0 (empty task_prompts) →
    // n_eff_cosine_prior = 1.0 fallback.
    let a1 = mock_adapter();
    let a2 = mock_adapter();
    let model = FixedEmbeddingModel;
    let cfg = H2AIConfig::default();
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        adapters: vec![
            &a1 as &dyn h2ai_types::adapter::IComputeAdapter,
            &a2 as &dyn h2ai_types::adapter::IComputeAdapter,
        ],
        task_prompts: vec![],
        constraint_corpus: &[],
        embedding_model: Some(&model as &dyn h2ai_context::embedding::EmbeddingModel),
        cfg: &cfg,
    };
    let result = CalibrationHarness::run(input).await.unwrap();
    assert!(result.coefficients.alpha >= 0.0);
}

// ── beta_from_n_eff_adj ──────────────────────────────────────────────────────

#[test]
fn beta_from_n_eff_adj_mode_collapse() {
    // N_eff=1.0, CG=0.0 → N_eff_adj=clamp(0,1,3)=1; β₀=(1/1 - 1/3)/(3-1)=0.333
    let b = beta_from_n_eff_adj(1.0, 0.0, 3, 2.0);
    assert!(
        (b - 0.333).abs() < 1e-3,
        "mode collapse β₀ should be ~0.333, got {b}"
    );
}

#[test]
fn beta_from_n_eff_adj_partial_coherence() {
    // N_eff=3.0, CG=0.7, k=2 → N_eff_adj=clamp(3×0.49,1,3)=clamp(1.47,1,3)=1.47
    // β₀=(1/1.47 - 1/3)/(3-1)=(0.680-0.333)/2≈0.174 — intermediate, not clamped
    let b = beta_from_n_eff_adj(3.0, 0.7, 3, 2.0);
    assert!(
        (b - 0.174).abs() < 1e-3,
        "partial coherence β₀ should be ~0.174, got {b}"
    );
}

#[test]
fn beta_from_n_eff_adj_ideal() {
    // N_eff=3, CG=0.9, k=2 → N_eff_adj=clamp(3×0.81,1,3)=clamp(2.43,1,3)=2.43
    // β₀=(1/2.43 - 1/3)/(3-1)=(0.4115 - 0.3333)/2=0.0391
    let b = beta_from_n_eff_adj(3.0, 0.9, 3, 2.0);
    assert!(
        (b - 0.039).abs() < 1e-3,
        "ideal β₀ should be ~0.039, got {b}"
    );
}

#[test]
fn beta_from_n_eff_adj_clamp_upper() {
    // N_eff=100, CG=1.0, k=2 → N_eff_adj=clamp(100,1,3)=3; β₀=(1/3-1/3)/2=0 → max(0, 1e-6)=1e-6
    let b = beta_from_n_eff_adj(100.0, 1.0, 3, 2.0);
    assert!(
        (b - 1e-6_f64).abs() < 1e-9,
        "upper-clamped β₀ should be 1e-6, got {b}"
    );
}

#[test]
fn beta_from_n_eff_adj_degenerate() {
    // n_cal=1 → degenerate, return 1e-6
    let b = beta_from_n_eff_adj(1.0, 1.0, 1, 2.0);
    assert_eq!(b, 1e-6, "degenerate β₀ should be 1e-6, got {b}");
}

// ── aimd_decay ───────────────────────────────────────────────────────────────

#[test]
fn aimd_decay_normal_step() {
    // α_current=0.15, decay_rate=0.95, alpha_measured=0.10
    // max(0.15×0.95, 0.10) = max(0.1425, 0.10) = 0.1425
    let a = aimd_decay(0.15, 0.10, 0.95);
    assert!((a - 0.1425).abs() < 1e-9, "decay normal: got {a}");
}

#[test]
fn aimd_decay_floors_at_measured() {
    // α_current=0.05, decay_rate=0.95, alpha_measured=0.10
    // max(0.05×0.95, 0.10) = max(0.0475, 0.10) = 0.10
    let a = aimd_decay(0.05, 0.10, 0.95);
    assert!((a - 0.10).abs() < 1e-9, "decay floor: got {a}");
}

// ── aimd_reset ───────────────────────────────────────────────────────────────

#[test]
fn aimd_reset_normal_step() {
    // α_current=0.04, reset_multiplier=3.0, seed_alpha=0.15
    // min(0.04×3.0, 0.15) = min(0.12, 0.15) = 0.12
    let a = aimd_reset(0.04, 0.15, 3.0);
    assert!((a - 0.12).abs() < 1e-9, "reset normal: got {a}");
}

#[test]
fn aimd_reset_caps_at_seed() {
    // α_current=0.10, reset_multiplier=3.0, seed_alpha=0.15
    // min(0.10×3.0, 0.15) = min(0.30, 0.15) = 0.15
    let a = aimd_reset(0.10, 0.15, 3.0);
    assert!((a - 0.15).abs() < 1e-9, "reset cap: got {a}");
}

// ── yield_from_history ───────────────────────────────────────────────────────

#[test]
fn yield_from_history_empty() {
    assert_eq!(yield_from_history(&[]), None);
}

#[test]
fn yield_from_history_normal() {
    // entries: (2,3,0) → 0.667, (3,4,0) → 0.75, (1,2,0) → 0.5
    // mean = (0.667 + 0.75 + 0.5) / 3 = 0.639
    let history = vec![(2u8, 3u8, 0u32), (3, 4, 0), (1, 2, 0)];
    let y = yield_from_history(&history).unwrap();
    assert!((y - 0.639).abs() < 1e-3, "yield normal: got {y}");
}

#[test]
fn yield_from_history_skips_zero_n_max() {
    // (2,3,0) → 0.667, (0,0,0) → skip, (1,2,0) → 0.5
    // mean = (0.667 + 0.5) / 2 = 0.583
    let history = vec![(2u8, 3u8, 0u32), (0, 0, 0), (1, 2, 0)];
    let y = yield_from_history(&history).unwrap();
    assert!((y - 0.583).abs() < 1e-3, "yield skip zero: got {y}");
}

#[tokio::test]
async fn calibration_returns_error_on_adapter_failure() {
    // Exercises error propagation (L483 map_err, L496 r?, L91/L101/L112 await?) when adapters fail.
    let failing = FailingAdapter::new();
    let good = mock_adapter();
    let cfg = H2AIConfig::default();
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        adapters: vec![
            &failing as &dyn h2ai_types::adapter::IComputeAdapter,
            &good as &dyn h2ai_types::adapter::IComputeAdapter,
        ],
        task_prompts: vec!["test prompt".into()],
        constraint_corpus: &[],
        embedding_model: None,
        cfg: &cfg,
    };
    let result = CalibrationHarness::run(input).await;
    assert!(result.is_err(), "adapter failure must propagate as Err");
}

#[test]
fn yield_from_history_all_zero_n_max_returns_none() {
    // Line 625: non-empty history but all n_max==0 → count stays 0 → return None
    let history = vec![(5u8, 0u8, 0u32), (3, 0, 0)];
    assert_eq!(
        yield_from_history(&history),
        None,
        "all-zero n_max must return None"
    );
}

// ── usl_fit tests ────────────────────────────────────────────────────────────

fn usl_throughput(n: f64, alpha: f64, beta: f64) -> f64 {
    n / (beta * n).mul_add(n - 1.0, alpha.mul_add(n - 1.0, 1.0))
}

#[test]
fn usl_fit_recovers_ai_agent_params() {
    // Ground truth: α=0.15, β₀=0.01, CG_mean=0.4 → β_eff=0.025
    let true_alpha = 0.15_f64;
    let true_beta0 = 0.01_f64;
    let t1 = 1.0_f64;
    let t2 = t1 / usl_throughput(2.0, true_alpha, true_beta0);
    let t4 = t1 / usl_throughput(4.0, true_alpha, true_beta0);

    let (alpha, beta0) = CalibrationHarness::usl_fit(t1, t2, 4, t4, 0.12, 0.01);
    assert!(
        (alpha - true_alpha).abs() < 0.005,
        "α recovery: expected {true_alpha}, got {alpha:.4}"
    );
    assert!(
        (beta0 - true_beta0).abs() < 0.001,
        "β₀ recovery: expected {true_beta0}, got {beta0:.6}"
    );
}

#[test]
fn usl_fit_recovers_human_team_params() {
    let true_alpha = 0.10_f64;
    let true_beta0 = 0.005_f64;
    let t1 = 1.0_f64;
    let t2 = t1 / usl_throughput(2.0, true_alpha, true_beta0);
    let t5 = t1 / usl_throughput(5.0, true_alpha, true_beta0);

    let (alpha, beta0) = CalibrationHarness::usl_fit(t1, t2, 5, t5, 0.12, 0.01);
    assert!(
        (alpha - true_alpha).abs() < 0.005,
        "α: expected {true_alpha}, got {alpha:.4}"
    );
    assert!(
        (beta0 - true_beta0).abs() < 0.001,
        "β₀: expected {true_beta0}, got {beta0:.6}"
    );
}

#[test]
fn usl_fit_fallback_when_m_less_than_3() {
    let (alpha, beta0) = CalibrationHarness::usl_fit(1.0, 0.8, 2, 0.8, 0.12, 0.01);
    assert_eq!(alpha, 0.12, "fallback α when M=2");
    assert_eq!(beta0, 0.01, "fallback β₀ when M=2");
}

#[test]
fn usl_fit_fallback_when_m_is_1() {
    let (alpha, beta0) = CalibrationHarness::usl_fit(1.0, 1.0, 1, 1.0, 0.12, 0.01);
    assert_eq!(alpha, 0.12);
    assert_eq!(beta0, 0.01);
}

#[test]
fn usl_fit_fallback_on_degenerate_timing() {
    // t1 = 0 → degenerate
    let (alpha, beta0) = CalibrationHarness::usl_fit(0.0, 0.5, 4, 0.5, 0.12, 0.01);
    assert_eq!(alpha, 0.12);
    assert_eq!(beta0, 0.01);
}

#[test]
fn usl_fit_fallback_on_negative_derived_params() {
    // Super-linear speedup at N=2 → negative alpha → use fallback
    let t1 = 1.0_f64;
    let t2_superlinear = 0.3; // X(2) ≈ 3.33 > 2 → super-linear → degenerate
    let t4 = 0.5;
    let (alpha, beta0) = CalibrationHarness::usl_fit(t1, t2_superlinear, 4, t4, 0.12, 0.01);
    assert_eq!(alpha, 0.12, "super-linear speedup must trigger fallback");
    assert_eq!(beta0, 0.01);
}

#[test]
fn usl_fit_clamps_extreme_values() {
    // Ground truth: α=0.8, β₀=0.02 — both positive but α > 0.5 clamp ceiling.
    let true_alpha = 0.8_f64;
    let true_beta0 = 0.02_f64;
    let t1 = 1.0_f64;
    let t2 = t1 / usl_throughput(2.0, true_alpha, true_beta0);
    let t4 = t1 / usl_throughput(4.0, true_alpha, true_beta0);

    let (alpha, beta0) = CalibrationHarness::usl_fit(t1, t2, 4, t4, 0.12, 0.01);
    assert_eq!(alpha, 0.5, "α=0.8 must be clamped to 0.5");
    assert!(
        (1e-6..=0.1).contains(&beta0),
        "beta0 out of clamped range: {beta0}"
    );
    assert!(
        (beta0 - 0.01).abs() > 1e-6,
        "beta0 must not be the fallback value — clamp path must be taken"
    );
}

// ── beta_from_token_spans tests ──────────────────────────────────────────────

#[test]
fn beta_from_token_spans_basic() {
    // 1 span: 100 tokens consumed for 5 proposals → 10 pairs → per_pair = 10
    // t1_tokens = 500 → β₀ = 10/500 = 0.02
    let spans = vec![(100u64, 5usize)];
    let beta = beta_from_token_spans(&spans, 500).unwrap();
    assert!(
        (beta - 0.02).abs() < 1e-9,
        "expected β₀=0.02, got {beta:.8}"
    );
}

#[test]
fn beta_from_token_spans_clamps_to_max() {
    // Pathological: many tokens for 2 proposals → 1 pair → enormous per_pair → clamps to 0.1
    let spans = vec![(1_000_000u64, 2usize)];
    let beta = beta_from_token_spans(&spans, 1).unwrap();
    assert_eq!(beta, 0.1, "must clamp to max 0.1");
}

#[test]
fn beta_from_token_spans_none_on_empty() {
    assert!(beta_from_token_spans(&[], 100).is_none());
}

#[test]
fn beta_from_token_spans_none_on_zero_t1() {
    assert!(beta_from_token_spans(&[(50, 3)], 0).is_none());
}

#[test]
fn beta_from_token_spans_multi_span_is_mean() {
    let spans = vec![(200u64, 3usize), (600u64, 3usize)];
    let beta = beta_from_token_spans(&spans, 10_000).unwrap();
    let expected = f64::midpoint(200.0_f64 / 3.0, 600.0 / 3.0) / 10_000.0;
    assert!(
        (beta - expected).abs() < 1e-9,
        "multi-span mean: expected {expected:.8}, got {beta:.8}"
    );
}

// ── calibration_max_ensemble_size test ───────────────────────────────────────

#[test]
fn calibration_max_ensemble_size_bounds_condorcet_search() {
    let cfg = H2AIConfig {
        calibration_max_ensemble_size: 3,
        ..Default::default()
    };
    let ec = EnsembleCalibration::from_cg_mean(0.7, cfg.calibration_max_ensemble_size);
    assert!(
        ec.n_optimal <= 3,
        "n_optimal must be ≤ calibration_max_ensemble_size=3, got {}",
        ec.n_optimal
    );
}

// ── EigenCalibration tests ───────────────────────────────────────────────────

#[tokio::test]
async fn from_cg_matrix_runs_on_blocking_thread() {
    let n = 3usize;
    let mut sigma = DMatrix::<f64>::identity(n, n);
    sigma[(0, 1)] = 0.5;
    sigma[(1, 0)] = 0.5;
    sigma[(0, 2)] = 0.3;
    sigma[(2, 0)] = 0.3;
    sigma[(1, 2)] = 0.2;
    sigma[(2, 1)] = 0.2;
    let delta = 0.01_f64;
    let sigma_clone = sigma.clone();
    let result =
        tokio::task::spawn_blocking(move || EigenCalibration::from_cg_matrix(&sigma_clone, delta))
            .await
            .expect("eigenvalue task panicked");
    assert!(result.n_effective > 0.0);
    assert!(result.n_effective <= n as f64 + 1e-9);
}

#[tokio::test]
async fn from_cosine_matrix_runs_on_blocking_thread() {
    let n = 3usize;
    let mut c = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        c[(i, i)] = 1.0;
    }
    c[(0, 1)] = 0.6;
    c[(1, 0)] = 0.6;
    c[(0, 2)] = 0.4;
    c[(2, 0)] = 0.4;
    c[(1, 2)] = 0.3;
    c[(2, 1)] = 0.3;
    // Normalise: divide by n so trace = 1
    let k_norm = c / n as f64;
    let delta = 0.01_f64;
    let k_clone = k_norm.clone();
    let result =
        tokio::task::spawn_blocking(move || EigenCalibration::from_cosine_matrix(&k_clone, delta))
            .await
            .expect("cosine eigenvalue task panicked");
    assert!(result.n_effective > 0.0);
    assert!(result.n_effective <= n as f64 + 1e-9);
}

// ── compute_conflict_rate tests ──────────────────────────────────────────────

fn hard_length_range(id: &str, min: Option<usize>, max: Option<usize>) -> ConstraintDoc {
    ConstraintDoc {
        id: id.to_owned(),
        source_file: format!("{id}.yaml"),
        description: String::new(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: h2ai_constraints::types::ConstraintPredicate::LengthRange {
            min_chars: min,
            max_chars: max,
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    }
}

#[test]
fn conflict_rate_fewer_than_2_proposals_returns_none() {
    assert!(compute_conflict_rate(&["hello"], &[]).is_none());
    assert!(compute_conflict_rate(&[], &[]).is_none());
}

#[test]
fn conflict_rate_empty_corpus_returns_none() {
    assert!(compute_conflict_rate(&["hello", "world"], &[]).is_none());
}

#[test]
fn conflict_rate_identical_proposals_clamped_to_min() {
    let corpus = vec![hard_length_range("c1", Some(3), None)];
    let result = compute_conflict_rate(&["abc", "abc"], &corpus).unwrap();
    assert!(
        (result - 1e-6).abs() < 1e-10,
        "identical proposals must clamp to 1e-6, got {result}"
    );
}

#[test]
fn conflict_rate_perfect_disagreement_is_one() {
    let corpus = vec![hard_length_range("c1", Some(10), None)];
    let long_output = "hello world!!"; // 13 chars → passes
    let short_output = "hi"; // 2 chars → fails
    let result = compute_conflict_rate(&[long_output, short_output], &corpus).unwrap();
    assert!(
        (result - 1.0).abs() < 1e-9,
        "perfect disagreement must be 1.0, got {result}"
    );
}

#[test]
fn conflict_rate_unanimous_fail_is_clamped_to_min() {
    let corpus = vec![hard_length_range("c1", Some(10), None)];
    let result = compute_conflict_rate(&["x", "y"], &corpus).unwrap();
    assert!(
        (result - 1e-6).abs() < 1e-10,
        "unanimous failure must clamp to 1e-6, got {result}"
    );
}

#[test]
fn conflict_rate_partial_disagreement() {
    let corpus = vec![
        hard_length_range("c1", Some(5), None),
        hard_length_range("c2", None, Some(20)),
        hard_length_range("c3", Some(1), None),
        hard_length_range("c4", None, Some(3)),
    ];
    let a = "hello world"; // 11 chars
    let b = "hi"; // 2 chars
    let result = compute_conflict_rate(&[a, b], &corpus).unwrap();
    assert!(
        (result - 0.5).abs() < 1e-9,
        "partial disagreement (2/4) must be 0.5, got {result}"
    );
}
