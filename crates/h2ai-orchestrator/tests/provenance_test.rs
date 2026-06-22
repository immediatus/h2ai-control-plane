use h2ai_orchestrator::provenance::{
    DocumentConfidence, ProvenanceMap, ProvisionConfidence, ProvisionProvenance,
};
use h2ai_types::events::{CheckVerdict, CheckVerdictKind};

fn make_verdict(kind: CheckVerdictKind) -> CheckVerdict {
    CheckVerdict {
        index: 0,
        kind,
        text: "evidence".into(),
    }
}

#[test]
fn all_verified_provisions_yields_high_confidence() {
    let mut map = ProvenanceMap::new();
    map.add_provision(ProvisionProvenance {
        provision_label: "Section 1".into(),
        confidence: ProvisionConfidence::Verified,
        verdicts: vec![make_verdict(CheckVerdictKind::Present)],
        gap_ids: vec![],
    });
    map.add_provision(ProvisionProvenance {
        provision_label: "Section 2".into(),
        confidence: ProvisionConfidence::Verified,
        verdicts: vec![make_verdict(CheckVerdictKind::Present)],
        gap_ids: vec![],
    });
    assert!(matches!(
        map.document_confidence(),
        DocumentConfidence::High
    ));
}

#[test]
fn single_requires_review_provision_dominates_document_confidence() {
    let mut map = ProvenanceMap::new();
    map.add_provision(ProvisionProvenance {
        provision_label: "Section 1".into(),
        confidence: ProvisionConfidence::Verified,
        verdicts: vec![make_verdict(CheckVerdictKind::Present)],
        gap_ids: vec![],
    });
    map.add_provision(ProvisionProvenance {
        provision_label: "Section 2".into(),
        confidence: ProvisionConfidence::RequiresReview,
        verdicts: vec![make_verdict(CheckVerdictKind::Missing)],
        gap_ids: vec!["g1".into()],
    });
    assert!(matches!(
        map.document_confidence(),
        DocumentConfidence::RequiresReview
    ));
}

#[test]
fn unverified_dominates_all_other_states() {
    let mut map = ProvenanceMap::new();
    map.add_provision(ProvisionProvenance {
        provision_label: "S1".into(),
        confidence: ProvisionConfidence::AutoCorrected,
        verdicts: vec![],
        gap_ids: vec!["g1".into()],
    });
    map.add_provision(ProvisionProvenance {
        provision_label: "S2".into(),
        confidence: ProvisionConfidence::Unverified,
        verdicts: vec![],
        gap_ids: vec![],
    });
    assert!(matches!(
        map.document_confidence(),
        DocumentConfidence::Unverified
    ));
}

#[test]
fn empty_map_yields_unverified_confidence() {
    let map = ProvenanceMap::new();
    assert!(matches!(
        map.document_confidence(),
        DocumentConfidence::Unverified
    ));
}
