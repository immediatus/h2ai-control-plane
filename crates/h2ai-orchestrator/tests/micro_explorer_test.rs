use h2ai_orchestrator::gap_checkers::{
    Gap, GapKind, GapResolveContext, GapResolver, GapSeverity, GapSource,
};
use h2ai_orchestrator::gap_resolvers::micro_explorer::MicroExplorerResolver;
use h2ai_test_utils::{failing_adapter, mock_adapter};
use std::sync::Arc;

fn make_gap(kind: GapKind) -> Gap {
    Gap {
        id: "g1".into(),
        kind,
        severity: GapSeverity::High,
        description: "Missing clause".into(),
        affected_provisions: vec![],
        depends_on: None,
        source: GapSource::SelectionPruning,
    }
}

#[test]
fn micro_explorer_handles_missing_provision() {
    let resolver = MicroExplorerResolver::new(Arc::new(mock_adapter("patched")));
    assert!(resolver.handles(&GapKind::MissingProvision));
}

#[test]
fn micro_explorer_does_not_handle_coherence_conflict() {
    let resolver = MicroExplorerResolver::new(Arc::new(mock_adapter("")));
    assert!(!resolver.handles(&GapKind::InterProvisionConflict));
}

#[test]
fn micro_explorer_does_not_handle_uncertain_domain() {
    // UncertainDomain gaps must stay open as RequiresReview provisions.
    // This is the invariant that makes TaskContextSeeder gaps deterministically
    // survive recovery and keep document_confidence != High.
    let resolver = MicroExplorerResolver::new(Arc::new(mock_adapter("")));
    assert!(!resolver.handles(&GapKind::UncertainDomain));
}

#[test]
fn micro_explorer_handles_incomplete_provision() {
    let resolver = MicroExplorerResolver::new(Arc::new(mock_adapter("patched")));
    assert!(resolver.handles(&GapKind::IncompleteProvision));
}

#[tokio::test]
async fn micro_explorer_returns_patched_text_when_adapter_succeeds() {
    let adapter = Arc::new(mock_adapter(
        "### SECTION 1\nPatched liability cap content.",
    ));
    let resolver = MicroExplorerResolver::new(adapter);
    let ctx = GapResolveContext {
        gap: make_gap(GapKind::MissingProvision),
        resolved_output: Arc::new("Original text.".into()),
        verified_provision_list: vec!["Section 2".into()],
        constraint_text: "1. Liability cap...".into(),
        constraint_ids: vec!["CONSTRAINT-1".into()],
    };
    let result = resolver.resolve(ctx).await;
    assert_eq!(result.gap_id, "g1");
    assert!(result.patched_text.is_some());
}

#[tokio::test]
async fn micro_explorer_returns_none_patch_when_adapter_fails() {
    let adapter = Arc::new(failing_adapter());
    let resolver = MicroExplorerResolver::new(adapter);
    let ctx = GapResolveContext {
        gap: make_gap(GapKind::MissingProvision),
        resolved_output: Arc::new("text".into()),
        verified_provision_list: vec![],
        constraint_text: "".into(),
        constraint_ids: vec![],
    };
    let result = resolver.resolve(ctx).await;
    assert!(result.patched_text.is_none());
}
