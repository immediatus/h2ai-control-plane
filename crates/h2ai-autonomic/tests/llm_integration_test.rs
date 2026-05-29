//! Calibration theory tests — prove calibration invariants hold with mock adapters.
//!
//! All LLM calls are replaced with `MockAdapter` so these tests run without any
//! external service.  Real-LLM behaviour is validated by the adapter integration
//! tests in h2ai-adapters.
//!
//! What these tests prove:
//!   - `β_eff` = β₀ × (1 − CG) holds in computed output
//!   - `N_max` = √((`1−α)/β_eff`) holds in computed output
//!   - With no corpus, CG equals `cfg.calibration_cg_fallback`
//!   - With a corpus, CG is measured (not the fallback)

use h2ai_autonomic::calibration::{CalibrationHarness, CalibrationInput};
use h2ai_config::H2AIConfig;
use h2ai_constraints::types::ConstraintDoc;
use h2ai_test_utils::mock_adapter;
use h2ai_types::adapter::IComputeAdapter;
use h2ai_types::identity::TaskId;
use h2ai_types::sizing::CoherencyCoefficients;

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

    assert!((0.0..1.0).contains(&alpha), "α ∉ [0,1): {alpha}");
    assert!(beta_base > 0.0, "β₀ must be > 0: {beta_base}");
    assert!((0.0..=1.0).contains(&cg), "CG ∉ [0,1]: {cg}");
    assert!(beta_eff > 0.0, "β_eff must be > 0: {beta_eff}");
    assert!(n_max >= 1.0, "N_max must be ≥ 1: {n_max}");

    let expected_beta_eff = (beta_base * (1.0 - cg)).max(1e-6);
    let rel_err = (beta_eff - expected_beta_eff).abs() / expected_beta_eff;
    assert!(
        rel_err < 0.01,
        "β_eff = β₀×(1−CG) violated: got={beta_eff:.6} want={expected_beta_eff:.6} rel_err={rel_err:.4}"
    );

    let expected_n_max = ((1.0 - alpha) / beta_eff).sqrt();
    let n_max_err = (n_max - expected_n_max).abs();
    assert!(
        n_max_err < 1.0,
        "N_max = √((1−α)/β_eff) violated: got={n_max:.2} want={expected_n_max:.2} err={n_max_err:.2}"
    );

    assert!(
        (2.0..=50.0).contains(&n_max),
        "N_max outside [2,50]: {n_max}"
    );

    eprintln!("  ✓ β_eff formula holds  (rel_err={rel_err:.5})");
    eprintln!("  ✓ N_max formula holds  (err={n_max_err:.3})");
    eprintln!("  ✓ N_max in plausible range [2,50]");
}

/// Proves: with no constraint corpus, CG equals the configured fallback,
/// and all structural invariants hold for the computed coefficients.
///
/// With `MockAdapter` (no corpus): `adapter_pair_cg` returns `calibration_cg_fallback` directly,
/// so `CG_mean` == `calibration_cg_fallback`. USL timing with instant mock adapters uses
/// fallback α/β₀.
#[tokio::test]
async fn calibration_no_corpus_fallback_cg_invariants_hold() {
    let a1 = mock_adapter("JWT is a stateless token used for authentication.");
    let a2 = mock_adapter("JWT is a stateless token used for authentication.");
    let a3 = mock_adapter("JWT is a stateless token used for authentication.");
    let cfg = H2AIConfig::default();

    let event = CalibrationHarness::run(CalibrationInput {
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
    .expect("calibration must succeed with mock adapters");

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

/// Proves: with a constraint corpus, CG is measured from Hamming distance on
/// constraint-satisfaction fingerprints (not the fallback).
///
/// With two identical `MockAdapters` and `LlmJudge` constraints:
/// - `eval_sync` returns 1.0 (pass-through) for `LlmJudge` predicates
/// - Both fingerprints are identical → Hamming distance = 0.0 → CG = 0.0
/// - 0.0 ≠ `calibration_cg_fallback` (0.7) → corpus measurement was used
#[tokio::test]
async fn calibration_with_corpus_measures_real_cg() {
    let a1 = mock_adapter("Use stateless JWT tokens for API authentication.");
    let a2 = mock_adapter("Use stateless JWT tokens for API authentication.");
    let cfg = H2AIConfig::default();

    let corpus = vec![
        ConstraintDoc::new_llm_judge(
            "stateless",
            "The solution must be stateless and must not use server-side sessions.",
        ),
        ConstraintDoc::new_soft_llm_judge("jwt", "Prefer JWT tokens. Mention JWT explicitly."),
    ];

    let event = CalibrationHarness::run(CalibrationInput {
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
    .expect("calibration must succeed with mock adapters");

    let cg = event.coefficients.cg_mean();
    eprintln!("\n── CG from Hamming distance on constraint profiles ──");
    eprintln!("  CG_mean  = {cg:.4}");
    eprintln!("  Samples  = {:?}", event.coefficients.cg_samples);
    eprintln!(
        "  Fallback = {:.3}  (should NOT be used)",
        cfg.calibration_cg_fallback
    );

    assert!(
        (cg - cfg.calibration_cg_fallback).abs() > 0.001,
        "CG={cg:.3} equals fallback={:.3} — corpus was not used for measurement",
        cfg.calibration_cg_fallback
    );
    assert!(
        (0.0..=1.0).contains(&cg),
        "Measured CG must be in [0,1]: {cg}"
    );

    assert_theory_invariants(&event.coefficients, "2-adapter / mock corpus / measured CG");

    if cg == 0.0 {
        eprintln!("  ✓ CG=0.0 — identical mock outputs: both adapters agree on all constraints");
    } else {
        eprintln!("  ✓ CG={cg:.3} — constraint-profile diversity detected");
    }
    eprintln!("  ✓ CG measured from constraint fingerprints (not fallback)");
}
