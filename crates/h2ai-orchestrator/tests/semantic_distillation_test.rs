use h2ai_orchestrator::induction::algorithmic::{
    distill_archetype_priors, distill_decomposition_templates, distill_tension_patterns,
};
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::reasoning_checkpoint::{ArchetypeResult, TaskMetaState};
use h2ai_types::sizing::TaskQuadrant;

fn make_meta(
    constraint_tags: Vec<String>,
    tensions: Vec<String>,
    archetype_results: Vec<ArchetypeResult>,
    quadrant: Option<TaskQuadrant>,
    retry_count: u32,
) -> TaskMetaState {
    TaskMetaState {
        task_id: TaskId::new(),
        tenant_id: TenantId("t1".into()),
        resolved_at: 0,
        constraint_tags,
        domain: None,
        task_quadrant: quadrant,
        shared_understanding: "default understanding".to_string(),
        tensions,
        archetype_results,
        thinking_iterations: 1,
        retry_count,
        retry_context_that_resolved: None,
        tried_topologies: vec![],
        tau_values_that_converged: None,
        system_context_with_rubric_hash: 0,
        constraint_corpus_fingerprint: 0,
    }
}

fn make_archetype(name: &str, confidence: f64) -> ArchetypeResult {
    ArchetypeResult {
        name: name.to_string(),
        confidence,
    }
}

// ── distill_archetype_priors ──────────────────────────────────────────────────

#[test]
fn distill_archetype_priors_empty_metas_returns_empty() {
    let result = distill_archetype_priors(&[]);
    assert!(result.is_empty());
}

#[test]
fn distill_archetype_priors_single_archetype_single_task() {
    let metas = vec![make_meta(
        vec!["billing".to_string()],
        vec![],
        vec![make_archetype("DEVIL_ADVOCATE", 0.8)],
        None,
        0,
    )];
    let priors = distill_archetype_priors(&metas);
    assert_eq!(priors.len(), 1);
    let p = &priors[0];
    assert_eq!(p.archetype_name, "DEVIL_ADVOCATE");
    assert_eq!(p.sample_count, 1);
    assert!((p.net_confidence - 0.8).abs() < 1e-9);
    assert!(p.avoid_for_tags.is_empty(), "sample_count=1 < 3: no avoid");
}

#[test]
fn distill_archetype_priors_avoid_tags_populated_when_three_low_confidence() {
    // DEVIL_ADVOCATE scores 0.3 on 3 tasks with "billing" tag — must get avoid_for_tags
    let metas = vec![
        make_meta(
            vec!["billing".to_string()],
            vec![],
            vec![make_archetype("DEVIL_ADVOCATE", 0.3)],
            None,
            1,
        ),
        make_meta(
            vec!["billing".to_string()],
            vec![],
            vec![make_archetype("DEVIL_ADVOCATE", 0.2)],
            None,
            1,
        ),
        make_meta(
            vec!["billing".to_string()],
            vec![],
            vec![make_archetype("DEVIL_ADVOCATE", 0.25)],
            None,
            1,
        ),
    ];
    let priors = distill_archetype_priors(&metas);
    let da = priors
        .iter()
        .find(|p| p.archetype_name == "DEVIL_ADVOCATE")
        .unwrap();
    assert_eq!(da.sample_count, 3);
    assert!(
        da.net_confidence < 0.4,
        "net_confidence={}",
        da.net_confidence
    );
    assert!(
        da.avoid_for_tags.contains(&"billing".to_string()),
        "billing must be in avoid_for_tags"
    );
}

#[test]
fn distill_archetype_priors_no_avoid_when_sample_count_below_threshold() {
    // Only 2 tasks — sample_count < MIN_SAMPLE_COUNT_FOR_AVOID=3 → no avoid
    let metas = vec![
        make_meta(
            vec!["billing".to_string()],
            vec![],
            vec![make_archetype("DEVIL_ADVOCATE", 0.1)],
            None,
            1,
        ),
        make_meta(
            vec!["billing".to_string()],
            vec![],
            vec![make_archetype("DEVIL_ADVOCATE", 0.1)],
            None,
            1,
        ),
    ];
    let priors = distill_archetype_priors(&metas);
    let da = priors
        .iter()
        .find(|p| p.archetype_name == "DEVIL_ADVOCATE")
        .unwrap();
    assert_eq!(da.sample_count, 2);
    assert!(
        da.avoid_for_tags.is_empty(),
        "sample_count=2 < 3: avoid_for_tags must be empty"
    );
}

#[test]
fn distill_archetype_priors_multiple_archetypes_independent() {
    let metas = vec![make_meta(
        vec!["auth".to_string()],
        vec![],
        vec![
            make_archetype("STEELMAN", 0.9),
            make_archetype("DEVIL_ADVOCATE", 0.7),
        ],
        None,
        0,
    )];
    let priors = distill_archetype_priors(&metas);
    assert_eq!(priors.len(), 2);
    let names: Vec<&str> = priors.iter().map(|p| p.archetype_name.as_str()).collect();
    assert!(names.contains(&"STEELMAN"));
    assert!(names.contains(&"DEVIL_ADVOCATE"));
}

// ── distill_tension_patterns ──────────────────────────────────────────────────

#[test]
fn distill_tension_patterns_empty_metas_returns_empty() {
    let result = distill_tension_patterns(&[]);
    assert!(result.is_empty());
}

#[test]
fn distill_tension_patterns_no_tensions_returns_empty() {
    let metas = vec![make_meta(vec![], vec![], vec![], None, 0)];
    let result = distill_tension_patterns(&metas);
    assert!(result.is_empty());
}

#[test]
fn distill_tension_patterns_identical_strings_produce_one_cluster() {
    let metas = vec![
        make_meta(
            vec![],
            vec!["rate limit vs throughput tradeoff".to_string()],
            vec![],
            None,
            0,
        ),
        make_meta(
            vec![],
            vec!["rate limit vs throughput tradeoff".to_string()],
            vec![],
            None,
            0,
        ),
    ];
    let patterns = distill_tension_patterns(&metas);
    assert_eq!(patterns.len(), 1, "identical strings must cluster");
    assert_eq!(patterns[0].frequency, 2);
}

#[test]
fn distill_tension_patterns_distinct_strings_produce_separate_clusters() {
    let metas = vec![make_meta(
        vec![],
        vec![
            "rate limit vs throughput".to_string(),
            "authentication bypass risk".to_string(),
        ],
        vec![],
        None,
        0,
    )];
    let patterns = distill_tension_patterns(&metas);
    assert_eq!(
        patterns.len(),
        2,
        "unrelated strings must not cluster: {:?}",
        patterns
            .iter()
            .map(|p| &p.canonical_text)
            .collect::<Vec<_>>()
    );
}

#[test]
fn distill_tension_patterns_shingles_populated() {
    let metas = vec![make_meta(
        vec![],
        vec!["cache invalidation timing constraint".to_string()],
        vec![],
        None,
        0,
    )];
    let patterns = distill_tension_patterns(&metas);
    assert!(
        !patterns[0].shingles.is_empty(),
        "shingles must be pre-computed"
    );
}

// ── distill_decomposition_templates ──────────────────────────────────────────

#[test]
fn distill_decomposition_templates_empty_metas_returns_empty() {
    let result = distill_decomposition_templates(&[]);
    assert!(result.is_empty());
}

#[test]
fn distill_decomposition_templates_groups_same_quadrant_and_tags() {
    // Two metas with same quadrant and constraint_tags → one template
    let metas = vec![
        {
            let mut m = make_meta(
                vec!["auth".to_string(), "rate-limit".to_string()],
                vec![],
                vec![],
                Some(TaskQuadrant::Coverage),
                0,
            );
            m.shared_understanding = "JWT validation first.".to_string();
            m
        },
        {
            let mut m = make_meta(
                vec!["auth".to_string(), "rate-limit".to_string()],
                vec![],
                vec![],
                Some(TaskQuadrant::Coverage),
                0,
            );
            m.shared_understanding = "JWT validation first.".to_string();
            m
        },
    ];
    let templates = distill_decomposition_templates(&metas);
    assert_eq!(
        templates.len(),
        1,
        "same quadrant+tags must produce one template"
    );
    assert_eq!(
        templates[0].success_count, 2,
        "both tasks had retry_count=0"
    );
}

#[test]
fn distill_decomposition_templates_prefers_lowest_retry_count() {
    // Two metas, same quadrant+tags, retry_count 0 and 2
    // Template shared_understanding must come from retry_count=0 meta
    let metas = vec![
        {
            let mut m = make_meta(
                vec!["billing".to_string()],
                vec![],
                vec![],
                Some(TaskQuadrant::Precision),
                2,
            );
            m.shared_understanding = "high-retry understanding".to_string();
            m
        },
        {
            let mut m = make_meta(
                vec!["billing".to_string()],
                vec![],
                vec![],
                Some(TaskQuadrant::Precision),
                0,
            );
            m.shared_understanding = "zero-retry understanding".to_string();
            m
        },
    ];
    let templates = distill_decomposition_templates(&metas);
    assert_eq!(templates.len(), 1);
    assert_eq!(
        templates[0].shared_understanding, "zero-retry understanding",
        "lowest retry_count must be chosen"
    );
    assert_eq!(
        templates[0].success_count, 1,
        "only one meta with retry_count=0"
    );
}

#[test]
fn distill_decomposition_templates_different_quadrants_produce_separate_templates() {
    let metas = vec![
        {
            let mut m = make_meta(
                vec!["billing".to_string()],
                vec![],
                vec![],
                Some(TaskQuadrant::Precision),
                0,
            );
            m.shared_understanding = "precision understanding".to_string();
            m
        },
        {
            let mut m = make_meta(
                vec!["billing".to_string()],
                vec![],
                vec![],
                Some(TaskQuadrant::Coverage),
                0,
            );
            m.shared_understanding = "coverage understanding".to_string();
            m
        },
    ];
    let templates = distill_decomposition_templates(&metas);
    assert_eq!(
        templates.len(),
        2,
        "different quadrants must yield separate templates"
    );
}
