use chrono::Utc;
use h2ai_adapters::mock::MockAdapter;
use h2ai_constraints::types::aggregate_compliance_score;
use h2ai_orchestrator::verification::{new_eval_cache, VerificationInput, VerificationPhase};
use h2ai_types::config::{AdapterKind, VerificationConfig};
use h2ai_types::events::ProposalEvent;
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::TauValue;
use std::sync::Arc;

fn make_proposal(task_id: TaskId, text: &str) -> ProposalEvent {
    ProposalEvent {
        task_id,
        explorer_id: ExplorerId::new(),
        tau: TauValue::new(0.5).unwrap(),
        generation: 0,
        raw_output: text.into(),
        token_cost: 10,
        adapter_kind: AdapterKind::CloudGeneric {
            endpoint: "mock".into(),
            api_key_env: "NONE".into(),
            model: None,
        },
        timestamp: Utc::now(),
    }
}

#[tokio::test]
async fn verification_passes_high_score() {
    // evaluator returns score 0.85 — should pass with default threshold 0.45
    // empty corpus → fallback to __rubric__ Hard constraint; hard_passes at 0.85 >= 0.45;
    // aggregate_compliance_score returns 1.0 (no Soft constraints) → overall = 1.0
    let evaluator = MockAdapter::new(r#"{"score": 0.85, "reason": "good"}"#.into());
    let proposal = make_proposal(TaskId::new(), "My proposal text");

    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    })
    .await;

    assert_eq!(out.passed.len(), 1, "expected 1 passed proposal");
    assert_eq!(out.failed.len(), 0, "expected 0 failed proposals");
    let (_, results, _) = &out.passed[0];
    assert_eq!(
        results.len(),
        1,
        "expected 1 compliance result (fallback rubric)"
    );
    assert!(
        (results[0].score - 0.85).abs() < 1e-9,
        "expected raw score 0.85, got {}",
        results[0].score
    );
}

#[tokio::test]
async fn verification_fails_low_score() {
    // evaluator returns score 0.3 — should fail with explicit threshold 0.6
    let evaluator = MockAdapter::new(r#"{"score": 0.3, "reason": "missing constraints"}"#.into());
    let proposal = make_proposal(TaskId::new(), "Incomplete proposal");

    let config = VerificationConfig {
        threshold: 0.6,
        ..Default::default()
    };

    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config,
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    })
    .await;

    assert_eq!(out.passed.len(), 0, "expected 0 passed proposals");
    assert_eq!(out.failed.len(), 1, "expected 1 failed proposal");
    let (_, results, violations, _) = &out.failed[0];
    assert_eq!(results.len(), 1, "expected 1 compliance result");
    assert!(
        (results[0].score - 0.3).abs() < 1e-9,
        "expected score 0.3, got {}",
        results[0].score
    );
    assert!(!violations.is_empty(), "expected at least one violation");
    assert_eq!(violations[0].constraint_id, "__rubric__");
}

#[tokio::test]
async fn verification_parallel_multiple_proposals() {
    // 4 proposals all at score 0.85 — verify all 4 are collected via parallel execution
    let evaluator = MockAdapter::new(r#"{"score": 0.85, "reason": "good"}"#.into());
    let task_id = TaskId::new();
    let proposals = (0..4)
        .map(|i| make_proposal(task_id.clone(), &format!("Proposal {i}")))
        .collect::<Vec<_>>();

    let out = VerificationPhase::run(VerificationInput {
        proposals,
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    })
    .await;

    assert_eq!(out.passed.len(), 4, "expected all 4 proposals to pass");
    assert_eq!(out.failed.len(), 0, "expected 0 failed proposals");
    for (_, results, _) in &out.passed {
        assert_eq!(results.len(), 1);
        assert!(
            (results[0].score - 0.85).abs() < 1e-9,
            "expected score 0.85, got {}",
            results[0].score
        );
    }
}

#[tokio::test]
async fn verification_parse_error_neutral_score() {
    // When the evaluator returns non-JSON (e.g. model outputs prose), llm_score_raw
    // returns 0.7 (neutral) rather than 0.0. Rationale: a malformed response means we
    // failed to evaluate — not that the proposal is wrong. The 0.7 neutral score is above
    // the default hard threshold (0.45), so the proposal passes. This is intentional:
    // a formatting failure from a small model should not silently prune valid proposals.
    let evaluator = MockAdapter::new("not valid json".into());
    let proposal = make_proposal(TaskId::new(), "Some proposal");

    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    })
    .await;

    assert_eq!(
        out.passed.len(),
        1,
        "neutral parse-error score (0.7) should pass threshold (0.45)"
    );
    assert_eq!(out.failed.len(), 0);
    let (_, results, _) = &out.passed[0];
    assert_eq!(results.len(), 1);
    assert!(
        (results[0].score - 0.7).abs() < 1e-9,
        "expected neutral fallback score 0.7, got {}",
        results[0].score
    );
}

#[tokio::test]
async fn verification_evaluator_empty_output_neutral_score() {
    // Empty evaluator output (e.g. adapter timed out before writing) returns 0.7 neutral.
    // Same reasoning as parse error: absence of evaluation evidence ≠ proposal failure.
    let evaluator = MockAdapter::new(String::new());
    let proposal = make_proposal(TaskId::new(), "Some proposal");

    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    })
    .await;

    assert_eq!(
        out.passed.len(),
        1,
        "neutral empty-output score (0.7) should pass threshold (0.45)"
    );
    assert_eq!(out.failed.len(), 0);
    let (_, results, _) = &out.passed[0];
    assert!(
        (results[0].score - 0.7).abs() < 1e-9,
        "expected 0.7 neutral, got {}",
        results[0].score
    );
}

#[tokio::test]
async fn verification_score_exactly_at_threshold_passes() {
    // score >= threshold is the pass condition — equality must pass.
    // Hard threshold in __rubric__ fallback is 0.45; we set both config threshold and
    // ensure the LLM score equals the hard threshold so it just passes.
    let threshold = 0.45;
    let evaluator = MockAdapter::new(format!(
        r#"{{"score": {threshold}, "reason": "at threshold"}}"#
    ));
    let proposal = make_proposal(TaskId::new(), "Proposal at threshold");
    let config = VerificationConfig {
        threshold,
        ..Default::default()
    };
    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config,
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    })
    .await;
    assert_eq!(out.passed.len(), 1, "score exactly at threshold must pass");
    assert_eq!(out.failed.len(), 0);
}

#[tokio::test]
async fn verification_empty_proposals_returns_empty_output() {
    let evaluator = MockAdapter::new(r#"{"score": 0.9, "reason": "good"}"#.into());
    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![],
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    })
    .await;
    assert!(out.passed.is_empty());
    assert!(out.failed.is_empty());
}

#[tokio::test]
async fn verification_score_clamped_above_one() {
    // LLM returns score > 1.0 — must be clamped to 1.0 and still pass.
    let evaluator = MockAdapter::new(r#"{"score": 2.5, "reason": "way above"}"#.into());
    let proposal = make_proposal(TaskId::new(), "Test proposal");
    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    })
    .await;
    assert_eq!(out.passed.len(), 1, "clamped score must still pass");
    let (_, results, _) = &out.passed[0];
    assert!(
        (results[0].score - 1.0).abs() < 1e-9,
        "score 2.5 must be clamped to 1.0, got {}",
        results[0].score
    );
}

#[tokio::test]
async fn verification_all_proposals_fail_below_threshold() {
    // score 0.5 < threshold 0.8 → hard gate fails (Hard threshold 0.45 is actually passed,
    // but overall config threshold 0.8 > aggregate score 1.0? No — aggregate is 1.0 only
    // when no Soft constraints exist, and Hard gate passes. So overall=1.0 >= threshold 0.8
    // would pass. To force failure: use a threshold above 0.5 that the Hard gate logic fails.
    // Actually __rubric__ Hard threshold is hardcoded 0.45, so score 0.5 >= 0.45 → hard_passes.
    // aggregate_compliance_score (no Soft) = 1.0. overall = 1.0 >= 0.8 → PASSES!
    // So we need score < 0.45 (Hard threshold) to fail. Use score 0.3.
    let threshold = 0.5;
    let evaluator = MockAdapter::new(r#"{"score": 0.3, "reason": "below"}"#.into());
    let task_id = TaskId::new();
    let proposals: Vec<_> = (0..3)
        .map(|i| make_proposal(task_id.clone(), &format!("Proposal {i}")))
        .collect();
    let config = VerificationConfig {
        threshold,
        ..Default::default()
    };
    let out = VerificationPhase::run(VerificationInput {
        proposals,
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config,
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    })
    .await;
    assert_eq!(out.passed.len(), 0, "all below hard threshold must fail");
    assert_eq!(out.failed.len(), 3);
}

#[tokio::test]
async fn verification_aggregate_score_used_for_passed() {
    // Verify that aggregate_compliance_score is used correctly on passed proposals.
    // With empty corpus (→ Hard __rubric__) and score 0.9, aggregate = 1.0.
    let evaluator = MockAdapter::new(r#"{"score": 0.9, "reason": "good"}"#.into());
    let proposal = make_proposal(TaskId::new(), "Good proposal");
    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    })
    .await;
    assert_eq!(out.passed.len(), 1);
    let (_, results, _) = &out.passed[0];
    let agg = aggregate_compliance_score(results);
    // No Soft constraints → aggregate returns 1.0
    assert!(
        (agg - 1.0).abs() < 1e-9,
        "aggregate of Hard-only results should be 1.0, got {agg}"
    );
}

#[tokio::test]
async fn verification_json_schema_predicate_passes_valid_json() {
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
    let schema = serde_json::json!({"type": "object", "required": ["result"]});
    let doc = ConstraintDoc {
        id: "schema_check".into(),
        source_file: "test".into(),
        description: "output must be JSON with result field".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::JsonSchema { schema },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    };
    let evaluator = MockAdapter::new(r#"{"score": 0.9, "reason": "unused"}"#.into());
    let proposal = make_proposal(TaskId::new(), r#"{"result": "ok"}"#);

    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[doc],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    })
    .await;

    assert_eq!(
        out.passed.len(),
        1,
        "valid JSON output must pass JsonSchema"
    );
    assert!((out.passed[0].1[0].score - 1.0).abs() < 1e-9);
}

#[tokio::test]
async fn verification_length_range_predicate_rejects_long_output() {
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
    let doc = ConstraintDoc {
        id: "length_check".into(),
        source_file: "test".into(),
        description: "output must be ≤ 10 chars".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::LengthRange {
            min_chars: None,
            max_chars: Some(10),
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    };
    let evaluator = MockAdapter::new(r#"{"score": 0.9, "reason": "unused"}"#.into());
    let proposal = make_proposal(
        TaskId::new(),
        "this is definitely longer than ten characters",
    );

    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[doc],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    })
    .await;

    assert_eq!(out.failed.len(), 1, "output exceeding max_chars must fail");
    assert!((out.failed[0].1[0].score - 0.0).abs() < 1e-9);
}

#[tokio::test]
async fn verification_oracle_execution_unreachable_scores_zero() {
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
    // Point at a URI that will immediately refuse — oracle_timeout/request_failed → 0.0
    let doc = ConstraintDoc {
        id: "oracle_check".into(),
        source_file: "test".into(),
        description: "run test suite".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::OracleExecution {
            test_runner_uri: "http://127.0.0.1:19999/run".into(),
            test_suite: "suite.py".into(),
            timeout_secs: 1,
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    };
    let evaluator = MockAdapter::new(r#"{"score": 0.9, "reason": "unused"}"#.into());
    let proposal = make_proposal(TaskId::new(), "some output");

    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[doc],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    })
    .await;

    assert_eq!(
        out.failed.len(),
        1,
        "unreachable oracle must fail the constraint"
    );
    assert!((out.failed[0].1[0].score - 0.0).abs() < 1e-9);
}

#[tokio::test]
async fn eval_cache_reuses_score_for_similar_proposals() {
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
    use std::sync::atomic::AtomicU32;
    use std::sync::Arc as StdArc;

    // Count how many times the evaluator is called.
    // With 2 nearly-identical proposals and 1 constraint, the second should be a cache hit.
    let call_count = StdArc::new(AtomicU32::new(0));
    let call_count_clone = StdArc::clone(&call_count);

    // Use a MockAdapter-compatible string — the mock always returns the same response.
    // We use the default MockAdapter since it doesn't provide call counting, so instead
    // we verify the cache hit flag on the output.
    let evaluator = MockAdapter::new(r#"{"score": 0.9, "reason": "good"}"#.into());
    drop(call_count_clone); // unused in this simplified form

    let doc = ConstraintDoc {
        id: "test_constraint".into(),
        source_file: "test".into(),
        description: "test".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.45 },
        predicate: ConstraintPredicate::LlmJudge {
            rubric: "Is this a good proposal?".into(),
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    };

    let cache = new_eval_cache();
    let task_id = TaskId::new();

    // First proposal: unique text — cache miss, calls LLM.
    let first = VerificationPhase::run(VerificationInput {
        proposals: vec![make_proposal(task_id.clone(), "The cache should use DashMap for concurrent access and repetition::similarity for text comparison.")],
        constraint_corpus: std::slice::from_ref(&doc),
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: Arc::clone(&cache),
        consensus_passes: 1,
    })
    .await;
    assert_eq!(first.passed.len(), 1);
    let (_, _, first_cache_hit) = &first.passed[0];
    assert!(!first_cache_hit, "first proposal should not be a cache hit");

    // Second proposal: nearly identical text (>0.85 similarity) — should be a cache hit.
    let second = VerificationPhase::run(VerificationInput {
        proposals: vec![make_proposal(task_id.clone(), "The cache should use DashMap for concurrent access and repetition::similarity for text comparison!")],
        constraint_corpus: std::slice::from_ref(&doc),
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: Arc::clone(&cache),
        consensus_passes: 1,
    })
    .await;
    assert_eq!(second.passed.len(), 1);
    let (_, _, second_cache_hit) = &second.passed[0];
    assert!(
        *second_cache_hit,
        "second similar proposal should be a cache hit"
    );
    drop(call_count);
}

#[tokio::test]
async fn eval_cache_does_not_reuse_for_dissimilar_proposals() {
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};

    let evaluator = MockAdapter::new(r#"{"score": 0.9, "reason": "good"}"#.into());

    let doc = ConstraintDoc {
        id: "test_constraint".into(),
        source_file: "test".into(),
        description: "test".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.45 },
        predicate: ConstraintPredicate::LlmJudge {
            rubric: "Is this a good proposal?".into(),
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    };

    let cache = new_eval_cache();
    let task_id = TaskId::new();

    // First proposal.
    let first = VerificationPhase::run(VerificationInput {
        proposals: vec![make_proposal(
            task_id.clone(),
            "Implementation using DashMap and token-overlap similarity for the evaluation cache.",
        )],
        constraint_corpus: std::slice::from_ref(&doc),
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: Arc::clone(&cache),
        consensus_passes: 1,
    })
    .await;
    assert_eq!(first.passed.len(), 1);

    // Second proposal: completely different text — should NOT be a cache hit.
    let second = VerificationPhase::run(VerificationInput {
        proposals: vec![make_proposal(
            task_id.clone(),
            "A completely different approach using global variables and cosine embeddings.",
        )],
        constraint_corpus: &[doc],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: Arc::clone(&cache),
        consensus_passes: 1,
    })
    .await;
    assert_eq!(second.passed.len(), 1);
    let (_, _, second_cache_hit) = &second.passed[0];
    assert!(
        !second_cache_hit,
        "dissimilar proposal should not be a cache hit"
    );
}

#[test]
fn record_adversarial_comparison_defaults_false() {
    use h2ai_types::config::VerificationConfig;
    let cfg = VerificationConfig::default();
    assert!(!cfg.record_adversarial_comparison);
}

#[tokio::test]
async fn panel_single_variant_matches_run_outcome() {
    use h2ai_config::JudgePanelConfig;
    use h2ai_orchestrator::judge_panel::JudgePanel;

    // Use MockAdapter returning score 0.85 — same setup as verification_passes_high_score.
    let evaluator = MockAdapter::new(r#"{"score": 0.85, "reason": "good"}"#.into());
    let proposal = make_proposal(TaskId::new(), "My proposal text");

    let input = VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    };

    // Build a 1-variant panel (no additional adapters → PersonaOnly 3 variants, but
    // we construct manually to get exactly 1 variant for the shortcut path).
    let panel = JudgePanel {
        variants: vec![h2ai_orchestrator::judge_panel::RuntimeJudgeVariant {
            adapter: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
            persona: h2ai_types::judge::JudgePersona::Literal,
            temperature_override: None,
        }],
        diversity_kind: h2ai_types::judge::PanelDiversityKind::PersonaOnly,
    };

    let panel_cfg = JudgePanelConfig::default();
    let (out, uncertain_map) = VerificationPhase::run_with_panel(input, &panel, &panel_cfg).await;

    // Single-variant path delegates to run() — same pass/fail outcome as run().
    assert_eq!(out.passed.len(), 1, "expected 1 passed proposal");
    assert_eq!(out.failed.len(), 0, "expected 0 failed proposals");
    // Single-variant → no panel disagreement possible.
    assert!(
        uncertain_map.is_empty(),
        "single-variant panel must not produce uncertain constraints"
    );
}

#[tokio::test]
async fn panel_all_pass_uncertain_map_is_empty() {
    use h2ai_config::JudgePanelConfig;
    use h2ai_orchestrator::judge_panel::JudgePanel;

    // Score 0.85 → all constraints pass → no uncertain constraints in map.
    let evaluator = MockAdapter::new(r#"{"score": 0.85, "reason": "good"}"#.into());
    let task_id = TaskId::new();
    let proposals = vec![
        make_proposal(task_id.clone(), "First proposal text"),
        make_proposal(task_id.clone(), "Second proposal text"),
    ];

    let input = VerificationInput {
        proposals,
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    };

    // Single-variant panel: all pass → uncertain_map must be empty.
    let panel = JudgePanel {
        variants: vec![h2ai_orchestrator::judge_panel::RuntimeJudgeVariant {
            adapter: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
            persona: h2ai_types::judge::JudgePersona::Literal,
            temperature_override: None,
        }],
        diversity_kind: h2ai_types::judge::PanelDiversityKind::PersonaOnly,
    };

    let panel_cfg = JudgePanelConfig::default();
    let (out, uncertain_map) = VerificationPhase::run_with_panel(input, &panel, &panel_cfg).await;

    assert_eq!(out.passed.len(), 2, "both proposals should pass");
    assert_eq!(out.failed.len(), 0);
    assert!(
        uncertain_map.is_empty(),
        "all-pass verdicts must not populate uncertain_map"
    );
}
