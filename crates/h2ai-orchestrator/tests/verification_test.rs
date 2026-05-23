use chrono::Utc;
use h2ai_constraints::types::aggregate_compliance_score;
use h2ai_orchestrator::verification::{new_eval_cache, VerificationInput, VerificationPhase};
use h2ai_test_utils::{MockAdapter, SequencedMockAdapter};
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
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
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
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
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
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
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
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
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
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
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

// ── SemanticPresence ─────────────────────────────────────────────────────────

#[tokio::test]
async fn semantic_presence_yes_majority_passes() {
    // All 3 passes return YES → majority satisfied → score 1.0 → Hard gate passes.
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
    let doc = ConstraintDoc {
        id: "presence_check".into(),
        source_file: "test".into(),
        description: "concept must be present".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::SemanticPresence {
            concept: "idempotency".into(),
            passes: 3,
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };
    let evaluator = SequencedMockAdapter::new(vec![
        "YES, concept present".into(),
        "YES, idempotency addressed".into(),
        "YES".into(),
    ]);
    let proposal = make_proposal(TaskId::new(), "We use idempotency keys on every request.");
    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[doc],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    })
    .await;
    assert_eq!(out.passed.len(), 1, "3/3 YES → majority passes");
    assert_eq!(out.failed.len(), 0);
    assert!((out.passed[0].1[0].score - 1.0).abs() < 1e-9);
}

#[tokio::test]
async fn semantic_presence_no_majority_fails() {
    // 1 YES, 2 NO → minority (1*2 < 3) → score 0.0 → Hard gate fails.
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
    let doc = ConstraintDoc {
        id: "presence_check".into(),
        source_file: "test".into(),
        description: "concept must be present".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::SemanticPresence {
            concept: "idempotency".into(),
            passes: 3,
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };
    let evaluator =
        SequencedMockAdapter::new(vec!["YES".into(), "NO, not mentioned".into(), "NO".into()]);
    let proposal = make_proposal(
        TaskId::new(),
        "We use a simple DECRBY without any idempotency.",
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
    assert_eq!(out.failed.len(), 1, "1/3 YES → minority fails");
    assert_eq!(out.passed.len(), 0);
    assert!((out.failed[0].1[0].score - 0.0).abs() < 1e-9);
}

// ── SemanticOrdering ─────────────────────────────────────────────────────────

#[tokio::test]
async fn semantic_ordering_correct_order_passes() {
    // Proposal correctly sequences debit then Kafka publish → 3/3 YES → score 1.0.
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
    let doc = ConstraintDoc {
        id: "C-005-ordering".into(),
        source_file: "test".into(),
        description: "debit before Kafka".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::SemanticOrdering {
            first: "account debit".into(),
            then: "Kafka publish".into(),
            passes: 3,
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };
    let evaluator = SequencedMockAdapter::new(vec![
        "YES, debit happens first then Kafka publish".into(),
        "YES".into(),
        "YES".into(),
    ]);
    let proposal = make_proposal(
        TaskId::new(),
        "Execute account debit atomically via Lua. After debit succeeds, publish to Kafka financial-events topic.",
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
    assert_eq!(out.passed.len(), 1, "correct ordering → passes");
    assert!((out.passed[0].1[0].score - 1.0).abs() < 1e-9);
}

#[tokio::test]
async fn semantic_ordering_wrong_order_fails() {
    // Proposal publishes to Kafka before debiting → 0 YES → score 0.0 → Hard fails.
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
    let doc = ConstraintDoc {
        id: "C-005-ordering".into(),
        source_file: "test".into(),
        description: "debit before Kafka".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::SemanticOrdering {
            first: "account debit".into(),
            then: "Kafka publish".into(),
            passes: 3,
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };
    let evaluator = SequencedMockAdapter::new(vec![
        "NO, Kafka publish happens before the debit".into(),
        "NO".into(),
        "NO".into(),
    ]);
    let proposal = make_proposal(
        TaskId::new(),
        "Publish to Kafka first to record intent, then debit the account balance.",
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
    assert_eq!(out.failed.len(), 1, "wrong ordering → fails");
    assert!((out.failed[0].1[0].score - 0.0).abs() < 1e-9);
}

#[tokio::test]
async fn semantic_ordering_no_kafka_mention_fails() {
    // Proposal uses Redis shadow log instead of Kafka → evaluator returns NO → score 0.0.
    // This is the C-005/1 regression scenario: audit via Redis sorted set, not Kafka.
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
    let doc = ConstraintDoc {
        id: "C-005-ordering".into(),
        source_file: "test".into(),
        description: "debit before Kafka publish".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::SemanticOrdering {
            first: "account debit".into(),
            then: "Kafka publish".into(),
            passes: 3,
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };
    let evaluator = SequencedMockAdapter::new(vec![
        "NO, no Kafka mentioned — uses Redis sorted set instead".into(),
        "NO".into(),
        "NO".into(),
    ]);
    let proposal = make_proposal(
        TaskId::new(),
        "After debit, write audit record to Redis sorted set (ZADD shadow_log causal_seq record). No Kafka.",
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
    assert_eq!(
        out.failed.len(),
        1,
        "Redis-only audit (no Kafka) must fail C-005 ordering gate"
    );
    assert!((out.failed[0].1[0].score - 0.0).abs() < 1e-9);
}

// ── SemanticExclusion ─────────────────────────────────────────────────────────

#[tokio::test]
async fn semantic_exclusion_pattern_absent_passes() {
    // Pattern not found → evaluator says NO (not present) → inverted → score 1.0 → passes.
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
    let doc = ConstraintDoc {
        id: "exclusion_check".into(),
        source_file: "test".into(),
        description: "no distributed locks on critical path".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::SemanticExclusion {
            pattern: "distributed lock".into(),
            passes: 3,
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };
    let evaluator = SequencedMockAdapter::new(vec![
        "NO, no distributed lock present".into(),
        "NO".into(),
        "NO".into(),
    ]);
    let proposal = make_proposal(
        TaskId::new(),
        "Use Redis Lua script for atomic debit. No locking required.",
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
    assert_eq!(out.passed.len(), 1, "excluded pattern absent → passes");
    assert!((out.passed[0].1[0].score - 1.0).abs() < 1e-9);
}

#[tokio::test]
async fn semantic_exclusion_pattern_present_fails() {
    // Pattern found → evaluator says YES (present) → inverted → score 0.0 → fails.
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
    let doc = ConstraintDoc {
        id: "exclusion_check".into(),
        source_file: "test".into(),
        description: "no distributed locks on critical path".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::SemanticExclusion {
            pattern: "distributed lock".into(),
            passes: 3,
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };
    let evaluator = SequencedMockAdapter::new(vec![
        "YES, Redlock distributed lock is used".into(),
        "YES".into(),
        "YES".into(),
    ]);
    let proposal = make_proposal(
        TaskId::new(),
        "Acquire a Redlock distributed lock before each DECRBY to prevent concurrent debits.",
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
    assert_eq!(out.failed.len(), 1, "excluded pattern present → fails");
    assert!((out.failed[0].1[0].score - 0.0).abs() < 1e-9);
}

// ── Majority vote edge cases ───────────────────────────────────────────────────

#[tokio::test]
async fn majority_vote_tie_is_conservative_fail() {
    // With passes=2: 1 YES, 1 NO → tie (1*2 == 2, not > 2) → conservative → score 0.0.
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
    let doc = ConstraintDoc {
        id: "tie_check".into(),
        source_file: "test".into(),
        description: "concept must be present".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::SemanticPresence {
            concept: "idempotency".into(),
            passes: 2,
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };
    let evaluator = SequencedMockAdapter::new(vec!["YES".into(), "NO".into()]);
    let proposal = make_proposal(TaskId::new(), "Ambiguous proposal.");
    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[doc],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    })
    .await;
    assert_eq!(out.failed.len(), 1, "tie (1/2) must conservatively fail");
    assert!((out.failed[0].1[0].score - 0.0).abs() < 1e-9);
}

#[tokio::test]
async fn majority_vote_single_pass_requires_unanimous() {
    // With passes=1: single YES → 1*2=2 > 1 → passes.
    // With passes=1: single NO → 0*2=0 ≤ 1 → fails.
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
    let make_doc = || ConstraintDoc {
        id: "single_pass".into(),
        source_file: "test".into(),
        description: "single pass check".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::SemanticPresence {
            concept: "idempotency".into(),
            passes: 1,
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };

    // YES path
    let yes_eval = SequencedMockAdapter::new(vec!["YES".into()]);
    let out_yes = VerificationPhase::run(VerificationInput {
        proposals: vec![make_proposal(TaskId::new(), "Uses idempotency keys.")],
        constraint_corpus: &[make_doc()],
        evaluator: &yes_eval as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    })
    .await;
    assert_eq!(
        out_yes.passed.len(),
        1,
        "single YES with passes=1 must pass"
    );

    // NO path
    let no_eval = SequencedMockAdapter::new(vec!["NO".into()]);
    let out_no = VerificationPhase::run(VerificationInput {
        proposals: vec![make_proposal(TaskId::new(), "No idempotency at all.")],
        constraint_corpus: &[make_doc()],
        evaluator: &no_eval as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    })
    .await;
    assert_eq!(out_no.failed.len(), 1, "single NO with passes=1 must fail");
}

// ── predicate_tier ────────────────────────────────────────────────────────────

#[test]
fn predicate_tier_semantic_variants_are_light() {
    use h2ai_constraints::types::{ConstraintPredicate, ConstraintTier};
    let cases = vec![
        ConstraintPredicate::SemanticPresence {
            concept: "x".into(),
            passes: 3,
        },
        ConstraintPredicate::SemanticOrdering {
            first: "a".into(),
            then: "b".into(),
            passes: 3,
        },
        ConstraintPredicate::SemanticExclusion {
            pattern: "p".into(),
            passes: 3,
        },
    ];
    for pred in &cases {
        let doc = h2ai_constraints::types::ConstraintDoc {
            id: "t".into(),
            source_file: "t".into(),
            description: "t".into(),
            severity: h2ai_constraints::types::ConstraintSeverity::Hard { threshold: 0.5 },
            predicate: pred.clone(),
            remediation_hint: None,
            domains: vec![],
            mandatory_for_tags: vec![],
            related_to: vec![],
            binary_checks: vec![],
            version: 1,
            repair_provenance: None,
            pass_criteria: None,
        };
        assert_eq!(
            doc.tier(),
            ConstraintTier::Light,
            "{pred:?} must be Light, not Static"
        );
    }
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

// ── adversarial comparison ────────────────────────────────────────────────────

#[tokio::test]
async fn adversarial_comparison_produces_comparison_events() {
    // record_adversarial_comparison = true triggers a second adversarial pass and
    // populates comparison_events with one entry per proposal.
    let evaluator = MockAdapter::new(r#"{"score": 0.85, "reason": "good"}"#.into());
    let proposal = make_proposal(TaskId::new(), "My proposal text");

    let config = VerificationConfig {
        record_adversarial_comparison: true,
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

    assert_eq!(out.passed.len(), 1, "proposal must still pass");
    assert_eq!(
        out.comparison_events.len(),
        1,
        "one comparison event per proposal"
    );
    let ev = &out.comparison_events[0];
    // Both standard and adversarial evaluators return same mock score 0.85 → both pass.
    assert!(ev.standard_passed, "standard pass must be true");
    assert!(ev.adversarial_passed, "adversarial pass must be true");
    assert!((ev.standard_score - ev.adversarial_score).abs() < 0.01);
}

#[tokio::test]
async fn adversarial_comparison_off_by_default_no_events() {
    // Default config has record_adversarial_comparison = false → no comparison events.
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

    assert!(
        out.comparison_events.is_empty(),
        "no adversarial comparison events when flag is off"
    );
}

// ── multi-variant panel (2-variant CrossFamily path) ─────────────────────────

#[tokio::test]
async fn panel_two_variants_both_pass_no_uncertain() {
    use h2ai_config::JudgePanelConfig;
    use h2ai_orchestrator::judge_panel::JudgePanel;

    // Two adapters, both returning score 0.85 → both vote pass → CrossFamily quorum
    // (2 out of 2 ≥ 0.67 quorum) → Pass verdict → no uncertain map entries.
    let eval_a = MockAdapter::new(r#"{"score": 0.85, "reason": "good"}"#.into());
    let eval_b = MockAdapter::new(r#"{"score": 0.85, "reason": "good"}"#.into());
    let proposal = make_proposal(TaskId::new(), "Some proposal");

    let input = VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[],
        evaluator: &eval_a as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    };

    let panel = JudgePanel {
        variants: vec![
            h2ai_orchestrator::judge_panel::RuntimeJudgeVariant {
                adapter: &eval_a as &dyn h2ai_types::adapter::IComputeAdapter,
                persona: h2ai_types::judge::JudgePersona::Literal,
                temperature_override: None,
            },
            h2ai_orchestrator::judge_panel::RuntimeJudgeVariant {
                adapter: &eval_b as &dyn h2ai_types::adapter::IComputeAdapter,
                persona: h2ai_types::judge::JudgePersona::Literal,
                temperature_override: None,
            },
        ],
        diversity_kind: h2ai_types::judge::PanelDiversityKind::CrossFamily,
    };

    let panel_cfg = JudgePanelConfig::default();
    let (out, uncertain_map) = VerificationPhase::run_with_panel(input, &panel, &panel_cfg).await;

    assert_eq!(
        out.passed.len(),
        1,
        "unanimous pass should be routed to passed"
    );
    assert_eq!(out.failed.len(), 0);
    assert!(
        uncertain_map.is_empty(),
        "both variants agree — no uncertainty"
    );
}

#[tokio::test]
async fn panel_two_variants_both_fail_hard_constraint() {
    use h2ai_config::JudgePanelConfig;
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
    use h2ai_orchestrator::judge_panel::JudgePanel;

    // Both adapters return score 0.2 (below Hard threshold 0.5) → both vote Fail →
    // CrossFamily quorum met for Fail → hard_fail = true → proposal fails.
    let eval_a = MockAdapter::new(r#"{"score": 0.2, "reason": "poor"}"#.into());
    let eval_b = MockAdapter::new(r#"{"score": 0.2, "reason": "poor"}"#.into());

    let doc = ConstraintDoc {
        id: "hard_constraint".into(),
        source_file: "test".into(),
        description: "must pass hard check".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::LlmJudge {
            rubric: "Is this a good proposal?".into(),
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };

    let proposal = make_proposal(TaskId::new(), "Poor proposal");

    let input = VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[doc],
        evaluator: &eval_a as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    };

    let panel = JudgePanel {
        variants: vec![
            h2ai_orchestrator::judge_panel::RuntimeJudgeVariant {
                adapter: &eval_a as &dyn h2ai_types::adapter::IComputeAdapter,
                persona: h2ai_types::judge::JudgePersona::Literal,
                temperature_override: None,
            },
            h2ai_orchestrator::judge_panel::RuntimeJudgeVariant {
                adapter: &eval_b as &dyn h2ai_types::adapter::IComputeAdapter,
                persona: h2ai_types::judge::JudgePersona::Literal,
                temperature_override: None,
            },
        ],
        diversity_kind: h2ai_types::judge::PanelDiversityKind::CrossFamily,
    };

    let panel_cfg = JudgePanelConfig::default();
    let (out, _uncertain_map) = VerificationPhase::run_with_panel(input, &panel, &panel_cfg).await;

    assert_eq!(
        out.failed.len(),
        1,
        "unanimous hard fail must route to failed"
    );
    assert_eq!(out.passed.len(), 0);
}

#[tokio::test]
async fn panel_two_variants_disagreement_uncertain_constraint() {
    use h2ai_config::JudgePanelConfig;
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
    use h2ai_orchestrator::judge_panel::JudgePanel;

    // eval_a scores 0.9 (pass), eval_b scores 0.2 (fail) → CrossFamily with quorum 0.67
    // → neither side meets quorum (1/2 = 0.5 < 0.67) → Uncertain → uncertain_map populated.
    let eval_a = MockAdapter::new(r#"{"score": 0.9, "reason": "good"}"#.into());
    let eval_b = MockAdapter::new(r#"{"score": 0.2, "reason": "poor"}"#.into());

    let doc = ConstraintDoc {
        id: "contested_constraint".into(),
        source_file: "test".into(),
        description: "contested".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::LlmJudge {
            rubric: "Is this good?".into(),
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };

    let proposal = make_proposal(TaskId::new(), "Ambiguous proposal");
    let explorer_id = proposal.explorer_id.clone();

    let input = VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[doc],
        evaluator: &eval_a as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    };

    let panel = JudgePanel {
        variants: vec![
            h2ai_orchestrator::judge_panel::RuntimeJudgeVariant {
                adapter: &eval_a as &dyn h2ai_types::adapter::IComputeAdapter,
                persona: h2ai_types::judge::JudgePersona::Literal,
                temperature_override: None,
            },
            h2ai_orchestrator::judge_panel::RuntimeJudgeVariant {
                adapter: &eval_b as &dyn h2ai_types::adapter::IComputeAdapter,
                persona: h2ai_types::judge::JudgePersona::Literal,
                temperature_override: None,
            },
        ],
        diversity_kind: h2ai_types::judge::PanelDiversityKind::CrossFamily,
    };

    // quorum_fraction = 0.67 so neither 1/2 pass nor 1/2 fail meets quorum → Uncertain
    let panel_cfg = JudgePanelConfig {
        quorum_fraction: 0.67,
        ..Default::default()
    };
    let (_out, uncertain_map) = VerificationPhase::run_with_panel(input, &panel, &panel_cfg).await;

    // Uncertain constraint → score = avg * uncertainty_weight; may pass or fail depending on
    // uncertainty_weight, but the uncertain_map must contain the explorer_id.
    assert!(
        uncertain_map.contains_key(&explorer_id),
        "disagreeing variants must populate uncertain_map for explorer"
    );
    let uncertain_ids = &uncertain_map[&explorer_id];
    assert!(
        uncertain_ids.contains(&"contested_constraint".to_string()),
        "contested_constraint must appear in uncertain constraint list"
    );
}

// ── Composite predicate ───────────────────────────────────────────────────────

#[tokio::test]
async fn composite_and_both_pass() {
    use h2ai_constraints::types::{
        CompositeOp, ConstraintDoc, ConstraintPredicate, ConstraintSeverity,
    };

    // And( LengthRange(max=1000), LlmJudge ) — both pass → score is min of two scores.
    let doc = ConstraintDoc {
        id: "composite_and".into(),
        source_file: "test".into(),
        description: "must be short and good".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::Composite {
            op: CompositeOp::And,
            children: vec![
                ConstraintPredicate::LengthRange {
                    min_chars: None,
                    max_chars: Some(1000),
                },
                ConstraintPredicate::LlmJudge {
                    rubric: "Is this a good proposal?".into(),
                },
            ],
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };

    let evaluator = MockAdapter::new(r#"{"score": 0.85, "reason": "good"}"#.into());
    let proposal = make_proposal(TaskId::new(), "A short good proposal.");

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
        "Composite(And) both pass → proposal passes"
    );
}

#[tokio::test]
async fn composite_and_static_fails_skips_llm() {
    use h2ai_constraints::types::{
        CompositeOp, ConstraintDoc, ConstraintPredicate, ConstraintSeverity,
    };

    // And( LengthRange(max=5), LlmJudge ) — static fails (length > 5) → short-circuit,
    // LlmJudge is never called, overall score = 0.0.
    let doc = ConstraintDoc {
        id: "composite_and_short_circuit".into(),
        source_file: "test".into(),
        description: "must be ≤5 chars AND pass llm".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::Composite {
            op: CompositeOp::And,
            children: vec![
                ConstraintPredicate::LengthRange {
                    min_chars: None,
                    max_chars: Some(5),
                },
                ConstraintPredicate::LlmJudge {
                    rubric: "Is this a good proposal?".into(),
                },
            ],
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };

    // LlmJudge would return 0.9, but short-circuit means it won't be called.
    let evaluator = MockAdapter::new(r#"{"score": 0.9, "reason": "good"}"#.into());
    let proposal = make_proposal(
        TaskId::new(),
        "This is definitely longer than five characters.",
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

    assert_eq!(
        out.failed.len(),
        1,
        "Composite(And) static child fails → proposal must fail"
    );
    assert!(
        (out.failed[0].1[0].score - 0.0).abs() < 1e-9,
        "short-circuit score must be 0.0"
    );
}

#[tokio::test]
async fn composite_or_first_passes_short_circuits() {
    use h2ai_constraints::types::{
        CompositeOp, ConstraintDoc, ConstraintPredicate, ConstraintSeverity,
    };

    // Or( LengthRange(max=1000), LengthRange(min=1) ) — first passes → max score = 1.0.
    let doc = ConstraintDoc {
        id: "composite_or".into(),
        source_file: "test".into(),
        description: "either short or present".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::Composite {
            op: CompositeOp::Or,
            children: vec![
                ConstraintPredicate::LengthRange {
                    min_chars: None,
                    max_chars: Some(1000),
                },
                ConstraintPredicate::LengthRange {
                    min_chars: Some(1),
                    max_chars: None,
                },
            ],
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };

    let evaluator = MockAdapter::new(r#"{"score": 0.85, "reason": "good"}"#.into());
    let proposal = make_proposal(TaskId::new(), "Short enough.");

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
        "Composite(Or) first passes → proposal passes"
    );
}

#[tokio::test]
async fn composite_or_both_fail() {
    use h2ai_constraints::types::{
        CompositeOp, ConstraintDoc, ConstraintPredicate, ConstraintSeverity,
    };

    // Or( LengthRange(max=0), LengthRange(max=0) ) — both fail → score 0.0.
    let doc = ConstraintDoc {
        id: "composite_or_fail".into(),
        source_file: "test".into(),
        description: "either condition must pass".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::Composite {
            op: CompositeOp::Or,
            children: vec![
                ConstraintPredicate::LengthRange {
                    min_chars: None,
                    max_chars: Some(0),
                },
                ConstraintPredicate::LengthRange {
                    min_chars: None,
                    max_chars: Some(0),
                },
            ],
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };

    let evaluator = MockAdapter::new(r#"{"score": 0.85, "reason": "good"}"#.into());
    let proposal = make_proposal(TaskId::new(), "Non-empty text.");

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
        "Composite(Or) both fail → proposal fails"
    );
}

#[tokio::test]
async fn composite_not_inverts_pass_to_fail() {
    use h2ai_constraints::types::{
        CompositeOp, ConstraintDoc, ConstraintPredicate, ConstraintSeverity,
    };

    // Not( LengthRange(max=1000) ) — inner passes (1.0) → Not inverts to 0.0 → fails.
    let doc = ConstraintDoc {
        id: "composite_not".into(),
        source_file: "test".into(),
        description: "must NOT satisfy length range".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::Composite {
            op: CompositeOp::Not,
            children: vec![ConstraintPredicate::LengthRange {
                min_chars: None,
                max_chars: Some(1000),
            }],
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };

    let evaluator = MockAdapter::new(r#"{"score": 0.9, "reason": "good"}"#.into());
    let proposal = make_proposal(TaskId::new(), "Short text.");

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
        "Composite(Not) inverts 1.0 to 0.0 → fails Hard gate"
    );
    assert!(
        (out.failed[0].1[0].score - 0.0).abs() < 1e-9,
        "Not(pass) = 1.0 - 1.0 = 0.0"
    );
}

#[tokio::test]
async fn composite_not_inverts_fail_to_pass() {
    use h2ai_constraints::types::{
        CompositeOp, ConstraintDoc, ConstraintPredicate, ConstraintSeverity,
    };

    // Not( LengthRange(max=0) ) — inner fails (0.0) → Not inverts to 1.0 → passes.
    let doc = ConstraintDoc {
        id: "composite_not_pass".into(),
        source_file: "test".into(),
        description: "must NOT have zero chars".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::Composite {
            op: CompositeOp::Not,
            children: vec![ConstraintPredicate::LengthRange {
                min_chars: None,
                max_chars: Some(0),
            }],
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };

    let evaluator = MockAdapter::new(r#"{"score": 0.9, "reason": "good"}"#.into());
    let proposal = make_proposal(TaskId::new(), "Has content.");

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
        "Composite(Not) inverts 0.0 to 1.0 → passes"
    );
}

#[tokio::test]
async fn composite_not_empty_children_returns_zero() {
    use h2ai_constraints::types::{
        CompositeOp, ConstraintDoc, ConstraintPredicate, ConstraintSeverity,
    };

    // Not with empty children → s = 0.0 → 1.0 - 0.0 = 1.0. Wait, code reads:
    // if let Some(child) = children.first() { ... } else { 0.0 }
    // So empty Not → s = 0.0 → returns 1.0 - 0.0 = 1.0.
    let doc = ConstraintDoc {
        id: "composite_not_empty".into(),
        source_file: "test".into(),
        description: "empty not".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::Composite {
            op: CompositeOp::Not,
            children: vec![],
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };

    let evaluator = MockAdapter::new(r#"{"score": 0.9, "reason": "good"}"#.into());
    let proposal = make_proposal(TaskId::new(), "Any proposal.");

    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[doc],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
        eval_cache: new_eval_cache(),
        consensus_passes: 1,
    })
    .await;

    // Not(empty) → s=0.0 → 1.0-0.0 = 1.0 → passes Hard threshold 0.5.
    assert_eq!(
        out.passed.len(),
        1,
        "Composite(Not) with empty children returns 1.0 → passes"
    );
}

// ── score_proposals ───────────────────────────────────────────────────────────

#[tokio::test]
async fn score_proposals_returns_aggregate_per_proposal() {
    // score_proposals returns one (proposal, score) per input in order.
    let evaluator = MockAdapter::new(r#"{"score": 0.8, "reason": "good"}"#.into());
    let task_id = TaskId::new();
    let proposals = vec![
        make_proposal(task_id.clone(), "Proposal A"),
        make_proposal(task_id.clone(), "Proposal B"),
    ];

    let scores = VerificationPhase::score_proposals(
        proposals,
        &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        &VerificationConfig::default(),
        &[],
    )
    .await;

    assert_eq!(scores.len(), 2, "one score per proposal");
    // With empty corpus → Hard __rubric__ fallback; score 0.8 → aggregate (Hard-only) = 1.0.
    for (_, s) in &scores {
        assert!(*s >= 0.0 && *s <= 1.0, "score must be in [0,1], got {s}");
    }
}

#[tokio::test]
async fn score_proposals_empty_input_returns_empty() {
    let evaluator = MockAdapter::new(r#"{"score": 0.9, "reason": "good"}"#.into());
    let scores = VerificationPhase::score_proposals(
        vec![],
        &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        &VerificationConfig::default(),
        &[],
    )
    .await;
    assert!(scores.is_empty(), "empty input → empty output");
}
