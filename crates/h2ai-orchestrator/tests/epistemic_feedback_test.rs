use async_trait::async_trait;
use h2ai_orchestrator::epistemic_feedback::{run_epistemic_feedback_loop, FeedbackLoopParams};
use h2ai_orchestrator::gap_checkers::{
    Gap, GapCheckContext, GapChecker, GapKind, GapResolveContext, GapResolver, GapSeverity,
    GapSource, ResolutionResult,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

// ── helpers ──────────────────────────────────────────────────────────────────

fn make_gap(id: &str, kind: GapKind) -> Gap {
    Gap {
        id: id.into(),
        kind,
        severity: GapSeverity::High,
        description: format!("test gap {id}"),
        affected_provisions: vec![],
        depends_on: None,
        source: GapSource::SelectionPruning,
    }
}

fn base_params(static_gaps: Vec<Gap>, resolver: Arc<dyn GapResolver>) -> FeedbackLoopParams {
    FeedbackLoopParams {
        static_gaps,
        initial_output: "original".into(),
        coherence_checker: None,
        resolver,
        verified_provision_list: vec![],
        constraint_text: String::new(),
        constraint_ids: vec![],
        max_passes: 5,
    }
}

// ── mock GapChecker ───────────────────────────────────────────────────────────

struct CountingChecker {
    count: Arc<AtomicUsize>,
    response: Vec<Gap>,
}

impl CountingChecker {
    fn new(response: Vec<Gap>) -> (Arc<AtomicUsize>, Self) {
        let count = Arc::new(AtomicUsize::new(0));
        (count.clone(), Self { count, response })
    }
}

#[async_trait]
impl GapChecker for CountingChecker {
    async fn check(&self, _doc: &str, _ctx: &GapCheckContext) -> Vec<Gap> {
        self.count.fetch_add(1, Ordering::SeqCst);
        self.response.clone()
    }
}

// ── mock GapResolver ──────────────────────────────────────────────────────────

struct PatchingResolver(String);

#[async_trait]
impl GapResolver for PatchingResolver {
    fn handles(&self, kind: &GapKind) -> bool {
        matches!(
            kind,
            GapKind::MissingProvision | GapKind::IncompleteProvision
        )
    }
    async fn resolve(&self, ctx: GapResolveContext) -> ResolutionResult {
        ResolutionResult {
            gap_id: ctx.gap.id.clone(),
            patched_text: Some(self.0.clone()),
            score_delta: 1.0,
        }
    }
}

struct FailingResolver;

#[async_trait]
impl GapResolver for FailingResolver {
    fn handles(&self, kind: &GapKind) -> bool {
        matches!(
            kind,
            GapKind::MissingProvision | GapKind::IncompleteProvision
        )
    }
    async fn resolve(&self, ctx: GapResolveContext) -> ResolutionResult {
        ResolutionResult {
            gap_id: ctx.gap.id.clone(),
            patched_text: None,
            score_delta: 0.0,
        }
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn no_resolvable_gaps_exits_immediately() {
    // UncertainDomain is not handled by any standard resolver; loop must exit on the
    // first pass without calling the resolver.
    let params = base_params(
        vec![make_gap("g1", GapKind::UncertainDomain)],
        Arc::new(FailingResolver),
    );
    let result = run_epistemic_feedback_loop(params).await;

    assert_eq!(result.final_output, "original");
    assert!(result.closed_ids.is_empty());
    assert_eq!(result.open_gaps.len(), 1);
    assert_eq!(result.coherence_checks_run, 0);
}

#[tokio::test]
async fn missing_provision_gap_is_closed_in_first_pass() {
    let params = base_params(
        vec![make_gap("g1", GapKind::MissingProvision)],
        Arc::new(PatchingResolver("patched output".into())),
    );
    let result = run_epistemic_feedback_loop(params).await;

    assert_eq!(result.final_output, "patched output");
    assert_eq!(result.closed_ids, vec!["g1"]);
    assert!(result.open_gaps.is_empty());
}

#[tokio::test]
async fn coherence_check_runs_once_per_pass_while_progress_made() {
    // Pass 1: coherence runs (count=1), static gap g1 closes → continues.
    // Pass 2: coherence runs (count=2), no resolvable gaps remain → exits.
    let (count, checker) = CountingChecker::new(vec![]);
    let mut params = base_params(
        vec![make_gap("g1", GapKind::MissingProvision)],
        Arc::new(PatchingResolver("patched".into())),
    );
    params.coherence_checker = Some(Arc::new(checker));
    params.max_passes = 5;

    let result = run_epistemic_feedback_loop(params).await;

    assert_eq!(count.load(Ordering::SeqCst), 2);
    assert_eq!(result.closed_ids, vec!["g1"]);
}

#[tokio::test]
async fn coherence_check_disabled_runs_zero_times() {
    let params = base_params(
        vec![make_gap("g1", GapKind::MissingProvision)],
        Arc::new(PatchingResolver("patched".into())),
    );
    // coherence_checker is None by default in base_params
    let result = run_epistemic_feedback_loop(params).await;

    assert_eq!(result.coherence_checks_run, 0);
    assert_eq!(result.closed_ids, vec!["g1"]);
}

#[tokio::test]
async fn failing_resolver_exits_after_first_stuck_pass() {
    let mut params = base_params(
        vec![make_gap("g1", GapKind::MissingProvision)],
        Arc::new(FailingResolver),
    );
    params.max_passes = 10;

    let result = run_epistemic_feedback_loop(params).await;

    assert_eq!(result.final_output, "original");
    assert!(result.closed_ids.is_empty());
    // gap stays open
    assert_eq!(result.open_gaps.len(), 1);
}

#[tokio::test]
async fn uncertain_domain_gap_always_stays_open() {
    // One resolvable gap + one UncertainDomain gap; the resolvable one closes,
    // the UncertainDomain must remain in open_gaps.
    let params = base_params(
        vec![
            make_gap("g1", GapKind::MissingProvision),
            make_gap("g-uncertain", GapKind::UncertainDomain),
        ],
        Arc::new(PatchingResolver("patched".into())),
    );
    let result = run_epistemic_feedback_loop(params).await;

    assert!(result.closed_ids.contains(&"g1".to_string()));
    let open_ids: Vec<&str> = result.open_gaps.iter().map(|g| g.id.as_str()).collect();
    assert!(
        open_ids.contains(&"g-uncertain"),
        "UncertainDomain must remain open"
    );
    assert!(
        !open_ids.contains(&"g1"),
        "resolved gap must not be in open_gaps"
    );
}

#[tokio::test]
async fn open_gaps_includes_last_pass_coherence_conflicts() {
    // CoherenceChecker always returns 1 InterProvisionConflict (non-resolvable).
    // After resolving the static gap, the final coherence conflict from the last pass
    // must appear in open_gaps so the ProvenanceMap can annotate it RequiresReview.
    let conflict_gap = Gap {
        id: "coh-1".into(),
        kind: GapKind::InterProvisionConflict,
        severity: GapSeverity::Medium,
        description: "conflict".into(),
        affected_provisions: vec![],
        depends_on: None,
        source: GapSource::CoherenceCheck,
    };
    let (_, checker) = CountingChecker::new(vec![conflict_gap]);
    let mut params = base_params(
        vec![make_gap("g1", GapKind::MissingProvision)],
        Arc::new(PatchingResolver("patched".into())),
    );
    params.coherence_checker = Some(Arc::new(checker));

    let result = run_epistemic_feedback_loop(params).await;

    assert!(result.closed_ids.contains(&"g1".to_string()));
    let has_conflict = result
        .open_gaps
        .iter()
        .any(|g| matches!(g.kind, GapKind::InterProvisionConflict));
    assert!(
        has_conflict,
        "last-pass coherence conflict must appear in open_gaps"
    );
}

#[tokio::test]
async fn max_passes_zero_returns_input_unchanged() {
    let mut params = base_params(
        vec![make_gap("g1", GapKind::MissingProvision)],
        Arc::new(PatchingResolver("patched".into())),
    );
    params.max_passes = 0;

    let result = run_epistemic_feedback_loop(params).await;

    assert_eq!(result.final_output, "original");
    assert!(result.closed_ids.is_empty());
    assert_eq!(result.coherence_checks_run, 0);
    // open_gaps = the static gaps (never entered the loop)
    assert_eq!(result.open_gaps.len(), 1);
}
