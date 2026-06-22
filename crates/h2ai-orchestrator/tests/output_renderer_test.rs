use h2ai_orchestrator::output_renderer::render_output;
use h2ai_orchestrator::provenance::{ProvenanceMap, ProvisionConfidence, ProvisionProvenance};

fn map_with_confidence(conf: ProvisionConfidence) -> ProvenanceMap {
    let mut m = ProvenanceMap::new();
    m.add_provision(ProvisionProvenance {
        provision_label: "Section 1".into(),
        confidence: conf,
        verdicts: vec![],
        gap_ids: vec![],
    });
    m
}

#[test]
fn clean_mode_always_prepends_confidence_header() {
    let map = map_with_confidence(ProvisionConfidence::Verified);
    let output = render_output("body text", &map, "clean");
    assert!(
        output.starts_with("> **Epistemic Confidence:"),
        "header missing: {}",
        &output[..80.min(output.len())]
    );
    assert!(output.contains("body text"));
}

#[test]
fn clean_mode_high_confidence_shows_verified_label() {
    let map = map_with_confidence(ProvisionConfidence::Verified);
    let output = render_output("body", &map, "clean");
    assert!(
        output.contains("High") || output.contains("Verified"),
        "{}",
        output
    );
}

#[test]
fn clean_mode_does_not_include_inline_annotations() {
    let mut map = ProvenanceMap::new();
    map.add_provision(ProvisionProvenance {
        provision_label: "Section 1".into(),
        confidence: ProvisionConfidence::RequiresReview,
        verdicts: vec![],
        gap_ids: vec!["g1".into()],
    });
    let output = render_output("body text", &map, "clean");
    assert!(
        !output.contains("⚠"),
        "clean mode must not include inline annotations"
    );
}

#[test]
fn audit_mode_includes_inline_annotations_for_gaps() {
    let mut map = ProvenanceMap::new();
    map.add_provision(ProvisionProvenance {
        provision_label: "Section 1".into(),
        confidence: ProvisionConfidence::RequiresReview,
        verdicts: vec![],
        gap_ids: vec!["g1".into()],
    });
    let output = render_output("body text", &map, "audit");
    assert!(
        output.contains("⚠") || output.contains("RequiresReview") || output.contains("g1"),
        "audit mode should include gap annotation: {}",
        output
    );
}

#[test]
fn audit_mode_includes_epistemic_footer() {
    let map = map_with_confidence(ProvisionConfidence::AutoCorrected);
    let output = render_output("body", &map, "audit");
    assert!(output.contains("Epistemic"), "footer missing in audit mode");
}

#[test]
fn requires_review_confidence_in_clean_mode_still_shows_requires_review_label() {
    let map = map_with_confidence(ProvisionConfidence::RequiresReview);
    let output = render_output("body", &map, "clean");
    assert!(
        output.contains("RequiresReview") || output.contains("Requires Review"),
        "clean mode should show worst-case confidence in header: {}",
        output
    );
}

#[test]
fn passthrough_mode_returns_text_unchanged() {
    let map = map_with_confidence(ProvisionConfidence::Verified);
    let body = "the exact output text";
    let output = render_output(body, &map, "passthrough");
    assert_eq!(output, body, "passthrough must not modify the text");
}

#[test]
fn passthrough_mode_no_header_even_with_requires_review() {
    let map = map_with_confidence(ProvisionConfidence::RequiresReview);
    let body = "output text";
    let output = render_output(body, &map, "passthrough");
    assert_eq!(output, body);
    assert!(!output.contains("Epistemic"), "passthrough adds no header");
}
