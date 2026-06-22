use chrono::Utc;
use h2ai_orchestrator::induction::algorithmic::{rank_and_filter, AlgorithmicInductionWorker};
use h2ai_orchestrator::induction::{InductionContext, InductionScheduler};
use h2ai_types::memory::{RetryHintPattern, TenantMemoryStore};

fn make_store_with_patterns(patterns: Vec<RetryHintPattern>) -> TenantMemoryStore {
    TenantMemoryStore {
        tenant_id: "t1".to_string(),
        generated_at: Utc::now(),
        task_count_seen: patterns.len() as u64,
        retry_hint_patterns: patterns,
        archetype_priors: vec![],
        tension_patterns: vec![],
        decomposition_templates: vec![],
    }
}

#[tokio::test]
async fn returns_none_when_store_has_no_matching_patterns() {
    let store = make_store_with_patterns(vec![]);
    let worker = AlgorithmicInductionWorker::new(store);
    let ctx = InductionContext {
        tenant_id: "t1".to_string(),
        task_class_tags: vec!["billing".to_string()],
        violated_constraint_ids: vec!["C-005".to_string()],
    };
    let result = worker.run_retroactive(&ctx).await;
    assert!(result.is_none(), "empty store must yield None");
}

#[tokio::test]
async fn returns_patterns_matching_any_context_tag() {
    let store = make_store_with_patterns(vec![
        RetryHintPattern {
            trigger_tags: vec!["billing".to_string()],
            exit_reason_kind: "ZeroSurvival".to_string(),
            hint_text: "use append-only schema".to_string(),
            success_count: 2,
            attempt_count: 5,
        },
        RetryHintPattern {
            trigger_tags: vec!["auth".to_string()],
            exit_reason_kind: "ZeroSurvival".to_string(),
            hint_text: "use JWT".to_string(),
            success_count: 1,
            attempt_count: 3,
        },
    ]);
    let worker = AlgorithmicInductionWorker::new(store);
    let ctx = InductionContext {
        tenant_id: "t1".to_string(),
        task_class_tags: vec!["billing".to_string()],
        violated_constraint_ids: vec!["C-005".to_string()],
    };
    let result = worker.run_retroactive(&ctx).await.unwrap();
    assert_eq!(result.patterns.len(), 1);
    assert_eq!(result.patterns[0].hint_text, "use append-only schema");
}

#[tokio::test]
async fn patterns_sorted_by_success_rate_descending() {
    let store = make_store_with_patterns(vec![
        RetryHintPattern {
            trigger_tags: vec!["billing".to_string()],
            exit_reason_kind: "ZeroSurvival".to_string(),
            hint_text: "low rate hint".to_string(),
            success_count: 0,
            attempt_count: 10, // rate ≈ (0+2)/(10+10) = 0.1
        },
        RetryHintPattern {
            trigger_tags: vec!["billing".to_string()],
            exit_reason_kind: "ZeroSurvival".to_string(),
            hint_text: "high rate hint".to_string(),
            success_count: 8,
            attempt_count: 10, // rate ≈ (8+2)/(10+10) = 0.5
        },
    ]);
    let worker = AlgorithmicInductionWorker::new(store);
    let ctx = InductionContext {
        tenant_id: "t1".to_string(),
        task_class_tags: vec!["billing".to_string()],
        violated_constraint_ids: vec![],
    };
    let result = worker.run_retroactive(&ctx).await.unwrap();
    assert_eq!(
        result.patterns[0].hint_text, "high rate hint",
        "highest success_rate must come first"
    );
}

#[tokio::test]
async fn induction_result_trigger_tags_match_context_tags() {
    let store = make_store_with_patterns(vec![RetryHintPattern {
        trigger_tags: vec!["billing".to_string()],
        exit_reason_kind: "ZeroSurvival".to_string(),
        hint_text: "hint A".to_string(),
        success_count: 1,
        attempt_count: 3,
    }]);
    let worker = AlgorithmicInductionWorker::new(store);
    let ctx = InductionContext {
        tenant_id: "t1".to_string(),
        task_class_tags: vec!["billing".to_string(), "audit-log".to_string()],
        violated_constraint_ids: vec!["C-005".to_string()],
    };
    let result = worker.run_retroactive(&ctx).await.unwrap();
    // trigger_tags on result should be the context tags (billing + audit-log + C-005)
    // so that is_compatible_with works downstream
    assert!(result.trigger_tags.contains(&"billing".to_string()));
}

#[test]
fn rank_and_filter_returns_empty_for_no_patterns() {
    let ctx = InductionContext {
        tenant_id: "t".to_string(),
        task_class_tags: vec!["billing".to_string()],
        violated_constraint_ids: vec![],
    };
    let result = rank_and_filter(&[], &ctx);
    assert!(result.is_empty());
}

#[test]
fn rank_and_filter_excludes_non_overlapping_tags() {
    let patterns = vec![RetryHintPattern {
        trigger_tags: vec!["auth".to_string()],
        exit_reason_kind: "ZeroSurvival".to_string(),
        hint_text: "jwt hint".to_string(),
        success_count: 5,
        attempt_count: 6,
    }];
    let ctx = InductionContext {
        tenant_id: "t".to_string(),
        task_class_tags: vec!["billing".to_string()],
        violated_constraint_ids: vec![],
    };
    let result = rank_and_filter(&patterns, &ctx);
    assert!(
        result.is_empty(),
        "auth pattern must not match billing context"
    );
}

#[test]
fn rank_and_filter_orders_by_success_rate_descending() {
    let patterns = vec![
        RetryHintPattern {
            trigger_tags: vec!["billing".to_string()],
            exit_reason_kind: "ZeroSurvival".to_string(),
            hint_text: "low".to_string(),
            success_count: 0,
            attempt_count: 10,
        },
        RetryHintPattern {
            trigger_tags: vec!["billing".to_string()],
            exit_reason_kind: "ZeroSurvival".to_string(),
            hint_text: "high".to_string(),
            success_count: 8,
            attempt_count: 10,
        },
    ];
    let ctx = InductionContext {
        tenant_id: "t".to_string(),
        task_class_tags: vec!["billing".to_string()],
        violated_constraint_ids: vec![],
    };
    let result = rank_and_filter(&patterns, &ctx);
    assert_eq!(result[0].hint_text, "high");
}

#[test]
fn rank_and_filter_matches_violated_constraint_ids() {
    let patterns = vec![RetryHintPattern {
        trigger_tags: vec!["CONSTRAINT-005".to_string()],
        exit_reason_kind: "ZeroSurvival".to_string(),
        hint_text: "audit-log hint".to_string(),
        success_count: 2,
        attempt_count: 3,
    }];
    let ctx = InductionContext {
        tenant_id: "t".to_string(),
        task_class_tags: vec!["billing".to_string()],
        violated_constraint_ids: vec!["CONSTRAINT-005".to_string()],
    };
    let result = rank_and_filter(&patterns, &ctx);
    assert_eq!(
        result.len(),
        1,
        "pattern matching violated_constraint_id must be included"
    );
}

// ── GAP-G1 Phase 2: AlgorithmicInductionWorker semantic memory ────────────────

use h2ai_types::memory::{ArchetypePrior, DecompositionTemplate, TensionPattern};
use h2ai_types::reasoning_checkpoint::ArchetypeResult as CheckpointArchetypeResult;

fn make_store_with_priors(
    priors: Vec<ArchetypePrior>,
    tension_patterns: Vec<TensionPattern>,
    decomposition_templates: Vec<DecompositionTemplate>,
) -> TenantMemoryStore {
    TenantMemoryStore {
        tenant_id: "t1".to_string(),
        generated_at: Utc::now(),
        task_count_seen: 5,
        retry_hint_patterns: vec![],
        archetype_priors: priors,
        tension_patterns,
        decomposition_templates,
    }
}

#[tokio::test]
async fn load_semantic_memory_returns_none_when_store_empty() {
    let store = make_store_with_priors(vec![], vec![], vec![]);
    let worker = AlgorithmicInductionWorker::new(store);
    let result = worker.load_semantic_memory("t1").await;
    assert!(result.is_none(), "empty priors must yield None");
}

#[tokio::test]
async fn load_semantic_memory_returns_priors_from_primed_store() {
    let prior = ArchetypePrior {
        archetype_name: "STEELMAN".to_string(),
        domain_tags: vec!["auth".to_string()],
        net_confidence: 0.85,
        sample_count: 4,
        avoid_for_tags: vec![],
    };
    let store = make_store_with_priors(vec![prior], vec![], vec![]);
    let worker = AlgorithmicInductionWorker::new(store);
    let result = worker.load_semantic_memory("t1").await.unwrap();
    assert_eq!(result.archetype_priors.len(), 1);
    assert_eq!(result.archetype_priors[0].archetype_name, "STEELMAN");
}

#[tokio::test]
async fn run_distillation_cycle_returns_priors_from_metas() {
    use h2ai_types::identity::{TaskId, TenantId};
    use h2ai_types::TaskMetaState;

    let meta = TaskMetaState {
        task_id: TaskId::new(),
        tenant_id: TenantId("t1".into()),
        resolved_at: 0,
        constraint_tags: vec!["billing".to_string()],
        domain: None,
        task_quadrant: None,
        shared_understanding: "billing is core".to_string(),
        tensions: vec![],
        archetype_results: vec![CheckpointArchetypeResult {
            name: "FIRST_PRINCIPLES".to_string(),
            confidence: 0.9,
        }],
        thinking_iterations: 1,
        retry_count: 0,
        retry_context_that_resolved: None,
        tried_topologies: vec![],
        tau_values_that_converged: None,
        system_context_with_rubric_hash: 0,
        constraint_corpus_fingerprint: 0,
    };
    let store = make_store_with_priors(vec![], vec![], vec![]);
    let worker = AlgorithmicInductionWorker::new(store);
    let result = worker.run_distillation_cycle(&[meta], "t1").await;
    assert_eq!(result.archetype_priors.len(), 1);
    assert_eq!(
        result.archetype_priors[0].archetype_name,
        "FIRST_PRINCIPLES"
    );
    assert_eq!(result.decomposition_templates.len(), 1);
}

#[tokio::test]
async fn run_distillation_cycle_empty_metas_returns_empty() {
    let store = make_store_with_priors(vec![], vec![], vec![]);
    let worker = AlgorithmicInductionWorker::new(store);
    let result = worker.run_distillation_cycle(&[], "t1").await;
    assert!(result.is_empty());
}
