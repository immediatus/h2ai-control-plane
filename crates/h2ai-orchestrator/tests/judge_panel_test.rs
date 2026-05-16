use h2ai_orchestrator::judge_panel::{aggregate_votes, ConstraintVerdict};
use h2ai_types::judge::PanelDiversityKind;

// ── CrossFamily aggregation ───────────────────────────────────────────────────

#[test]
fn cross_family_3_of_3_pass() {
    let v = aggregate_votes(3, 0, &PanelDiversityKind::CrossFamily, 0.67);
    assert!(matches!(v, ConstraintVerdict::Pass));
}

#[test]
fn cross_family_3_of_3_fail() {
    let v = aggregate_votes(0, 3, &PanelDiversityKind::CrossFamily, 0.67);
    assert!(matches!(v, ConstraintVerdict::Fail));
}

#[test]
fn cross_family_2_of_3_pass_meets_quorum() {
    // quorum_fraction = 0.60 → ceil(3 * 0.60) = 2 → Pass
    let v = aggregate_votes(2, 1, &PanelDiversityKind::CrossFamily, 0.60);
    assert!(matches!(v, ConstraintVerdict::Pass));
}

#[test]
fn cross_family_1_of_3_pass_is_uncertain() {
    let v = aggregate_votes(1, 2, &PanelDiversityKind::CrossFamily, 0.67);
    assert!(matches!(v, ConstraintVerdict::Uncertain { .. }));
}

#[test]
fn cross_family_single_variant_pass() {
    let v = aggregate_votes(1, 0, &PanelDiversityKind::CrossFamily, 0.67);
    assert!(matches!(v, ConstraintVerdict::Pass));
}

#[test]
fn cross_family_1_of_2_split_is_uncertain() {
    let v = aggregate_votes(1, 1, &PanelDiversityKind::CrossFamily, 0.67);
    assert!(matches!(v, ConstraintVerdict::Uncertain { .. }));
}

// ── PersonaOnly aggregation ───────────────────────────────────────────────────

#[test]
fn persona_only_unanimous_pass() {
    let v = aggregate_votes(3, 0, &PanelDiversityKind::PersonaOnly, 0.67);
    assert!(matches!(v, ConstraintVerdict::Pass));
}

#[test]
fn persona_only_unanimous_fail() {
    let v = aggregate_votes(0, 3, &PanelDiversityKind::PersonaOnly, 0.67);
    assert!(matches!(v, ConstraintVerdict::Fail));
}

#[test]
fn persona_only_any_split_is_uncertain() {
    let v = aggregate_votes(2, 1, &PanelDiversityKind::PersonaOnly, 0.67);
    assert!(matches!(v, ConstraintVerdict::Uncertain { .. }));
}

#[test]
fn persona_only_1_of_2_split_is_uncertain() {
    let v = aggregate_votes(1, 1, &PanelDiversityKind::PersonaOnly, 0.67);
    assert!(matches!(v, ConstraintVerdict::Uncertain { .. }));
}

#[test]
fn zero_votes_is_uncertain() {
    let v = aggregate_votes(0, 0, &PanelDiversityKind::CrossFamily, 0.67);
    assert!(matches!(v, ConstraintVerdict::Uncertain { .. }));
}
