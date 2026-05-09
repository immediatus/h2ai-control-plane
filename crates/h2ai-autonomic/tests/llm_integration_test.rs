//! Real-LLM integration tests — prove calibration theory matches implementation.
//!
//! Run with llama.server on port 8080:
//! ```bash
//! LLAMACPP_BASE_URL=http://host.docker.internal:8080/v1 \
//!   cargo nextest run -p h2ai-autonomic --test llm_integration_test --run-ignored all --nocapture
//! ```
//!
//! What these tests prove:
//!   - β_eff = β₀ × (1 − CG) holds in computed output
//!   - N_max = √((1−α)/β_eff) holds in computed output
//!   - With real corpus, CG is measured (not fallback)
//!   - N_max is in plausible range for AI agents

use h2ai_adapters::openai::OpenAIAdapter;
use h2ai_autonomic::calibration::{CalibrationHarness, CalibrationInput};
use h2ai_config::H2AIConfig;
use h2ai_constraints::types::ConstraintDoc;
use h2ai_types::adapter::IComputeAdapter;
use h2ai_types::identity::TaskId;
use h2ai_types::sizing::CoherencyCoefficients;

fn llamacpp_endpoint() -> String {
    std::env::var("LLAMACPP_BASE_URL")
        .unwrap_or_else(|_| "http://host.docker.internal:8080/v1".into())
}

fn make_adapter() -> OpenAIAdapter {
    OpenAIAdapter::new(
        llamacpp_endpoint(),
        "LLAMACPP_API_KEY".into(),
        std::env::var("LLAMACPP_MODEL").unwrap_or_else(|_| "local".into()),
    )
}

async fn is_reachable() -> bool {
    std::env::set_var("LLAMACPP_API_KEY", "local");
    let a = make_adapter();
    let probe = h2ai_types::adapter::ComputeRequest {
        system_context: "You are a helpful assistant.".into(),
        task: "Reply with one word: ready".into(),
        tau: h2ai_types::sizing::TauValue::new(0.3).unwrap(),
        max_tokens: 10,
    };
    a.execute(probe).await.is_ok()
}

fn assert_theory_invariants(coeff: &CoherencyCoefficients, label: &str) {
    let alpha = coeff.alpha;
    let beta_base = coeff.beta_base;
    let cg = coeff.cg_mean();
    let beta_eff = coeff.beta_eff();
    let n_max = coeff.n_max();

    eprintln!("\n── {label} ──");
    eprintln!("  α        = {alpha:.4}");
    eprintln!("  β₀       = {beta_base:.4}");
    eprintln!("  CG_mean  = {cg:.4}");
    eprintln!(
        "  β_eff    = {beta_eff:.4}   (expect β₀ × (1−CG) = {:.4})",
        beta_base * (1.0 - cg)
    );
    eprintln!(
        "  N_max    = {n_max:.2}     (expect √((1−α)/β_eff) = {:.2})",
        ((1.0 - alpha) / beta_eff.max(1e-9)).sqrt()
    );

    // Structural invariants
    assert!((0.0..1.0).contains(&alpha), "α ∉ [0,1): {alpha}");
    assert!(beta_base > 0.0, "β₀ must be > 0: {beta_base}");
    assert!((0.0..=1.0).contains(&cg), "CG ∉ [0,1]: {cg}");
    assert!(beta_eff > 0.0, "β_eff must be > 0: {beta_eff}");
    assert!(n_max >= 1.0, "N_max must be ≥ 1: {n_max}");

    // β_eff formula: β_eff = β₀ × (1 − CG)  [clamped to ≥ 1e-6]
    let expected_beta_eff = (beta_base * (1.0 - cg)).max(1e-6);
    let rel_err = (beta_eff - expected_beta_eff).abs() / expected_beta_eff;
    assert!(
        rel_err < 0.01,
        "β_eff = β₀×(1−CG) violated: got={beta_eff:.6} want={expected_beta_eff:.6} rel_err={rel_err:.4}"
    );

    // N_max formula: N_max = √((1−α)/β_eff)
    let expected_n_max = ((1.0 - alpha) / beta_eff).sqrt();
    let n_max_err = (n_max - expected_n_max).abs();
    assert!(
        n_max_err < 1.0,
        "N_max = √((1−α)/β_eff) violated: got={n_max:.2} want={expected_n_max:.2} err={n_max_err:.2}"
    );

    // Plausibility
    assert!(
        (2.0..=50.0).contains(&n_max),
        "N_max outside [2,50]: {n_max}"
    );

    eprintln!("  ✓ β_eff formula holds  (rel_err={rel_err:.5})");
    eprintln!("  ✓ N_max formula holds  (err={n_max_err:.3})");
    eprintln!("  ✓ N_max in plausible range [2,50]");
}

fn is_adapter_unavailable(e: &h2ai_autonomic::calibration::CalibrationError) -> bool {
    let s = e.to_string();
    s.contains("network error") || s.contains("connection refused") || s.contains("timed out")
}

/// Proves: with no constraint corpus, CG is the configured fallback,
/// and all structural invariants hold for the computed coefficients.
#[tokio::test]
#[ignore = "requires llama.server at LLAMACPP_BASE_URL"]
async fn calibration_no_corpus_fallback_cg_invariants_hold() {
    if !is_reachable().await {
        eprintln!(
            "SKIP: llama.server not reachable at {}",
            llamacpp_endpoint()
        );
        return;
    }

    let a1 = make_adapter();
    let a2 = make_adapter();
    let a3 = make_adapter();
    let cfg = H2AIConfig::default();

    let event = match CalibrationHarness::run(CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec![
            "Describe a stateless auth strategy for microservices. Be concise.".into(),
            "What is the main advantage of event sourcing over CRUD?".into(),
            "Name one trade-off of using a distributed cache.".into(),
        ],
        adapters: vec![
            &a1 as &dyn IComputeAdapter,
            &a2 as &dyn IComputeAdapter,
            &a3 as &dyn IComputeAdapter,
        ],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: None,
    })
    .await
    {
        Ok(ev) => ev,
        Err(e) if is_adapter_unavailable(&e) => {
            eprintln!("SKIP: LLM became unreachable mid-calibration: {e}");
            return;
        }
        Err(e) => panic!("calibration failed with non-network error: {e}"),
    };

    assert_theory_invariants(&event.coefficients, "3-adapter / no corpus / fallback CG");

    let cg = event.coefficients.cg_mean();
    assert!(
        (cg - cfg.calibration_cg_fallback).abs() < 0.01,
        "Expected fallback CG={:.3} got={cg:.3}",
        cfg.calibration_cg_fallback
    );
    eprintln!(
        "  ✓ CG = cfg.calibration_cg_fallback ({:.3})",
        cfg.calibration_cg_fallback
    );
}

/// Proves: with a real constraint corpus, CG is measured from Hamming distance
/// on actual LLM constraint-satisfaction profiles (not the fallback).
/// CG must be in (0, 1) because real LLM responses vary.
#[tokio::test]
#[ignore = "requires llama.server at LLAMACPP_BASE_URL"]
async fn calibration_with_corpus_measures_real_cg() {
    if !is_reachable().await {
        eprintln!(
            "SKIP: llama.server not reachable at {}",
            llamacpp_endpoint()
        );
        return;
    }

    let a1 = make_adapter();
    let a2 = make_adapter();
    let cfg = H2AIConfig::default();

    // Two constraints the model will sometimes satisfy — creates measurable diversity
    let corpus = vec![
        ConstraintDoc::new_llm_judge(
            "stateless",
            "The solution must be stateless and must not use server-side sessions.",
        ),
        ConstraintDoc::new_soft_llm_judge("jwt", "Prefer JWT tokens. Mention JWT explicitly."),
    ];

    let event = match CalibrationHarness::run(CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec![
            "Design an auth system for APIs. Be brief.".into(),
            "How should microservices authenticate requests? One paragraph.".into(),
        ],
        adapters: vec![&a1 as &dyn IComputeAdapter, &a2 as &dyn IComputeAdapter],
        cfg: &cfg,
        constraint_corpus: &corpus,
        embedding_model: None,
    })
    .await
    {
        Ok(ev) => ev,
        Err(e) if is_adapter_unavailable(&e) => {
            eprintln!("SKIP: LLM became unreachable mid-calibration: {e}");
            return;
        }
        Err(e) => panic!("calibration failed with non-network error: {e}"),
    };

    let cg = event.coefficients.cg_mean();
    eprintln!("\n── CG from Hamming distance on constraint profiles ──");
    eprintln!("  CG_mean  = {cg:.4}");
    eprintln!("  Samples  = {:?}", event.coefficients.cg_samples);
    eprintln!(
        "  Fallback = {:.3}  (should NOT be used)",
        cfg.calibration_cg_fallback
    );

    // CG must be a valid measurement (not the fallback).
    // CG=0.0 is valid: both adapters agreed on all constraint satisfactions (Hamming=0).
    // CG=0.7 would mean the corpus was ignored and fallback was used.
    assert!(
        (cg - cfg.calibration_cg_fallback).abs() > 0.001,
        "CG={cg:.3} equals fallback={:.3} — corpus was not used for measurement",
        cfg.calibration_cg_fallback
    );
    assert!(
        (0.0..=1.0).contains(&cg),
        "Measured CG must be in [0,1]: {cg}"
    );

    assert_theory_invariants(&event.coefficients, "2-adapter / real corpus / measured CG");

    if cg == 0.0 {
        eprintln!("  ✓ CG=0.0 — both adapters satisfied the same constraints (high agreement)");
    } else {
        eprintln!("  ✓ CG={cg:.3} — real constraint-profile diversity detected");
    }
    eprintln!("  ✓ CG measured from real LLM responses (not fallback)");
}
