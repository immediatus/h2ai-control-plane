use h2ai_config::H2AIConfig;
use std::io::Write;

fn cfg_with_model_max_tokens(n: u64) -> H2AIConfig {
    let mut tmp = tempfile::NamedTempFile::with_suffix(".toml").unwrap();
    writeln!(tmp, "model_max_tokens = {n}").unwrap();
    H2AIConfig::load_layered(Some(tmp.path())).expect("valid config")
}

#[test]
fn thinking_loop_archetype_select_max_tokens_cascades_from_model() {
    let cfg = cfg_with_model_max_tokens(4096);
    assert_eq!(cfg.thinking_loop.archetype_select_max_tokens, 4096);
}

#[test]
fn thinking_loop_brainstorm_max_tokens_cascades_from_model() {
    let cfg = cfg_with_model_max_tokens(4096);
    assert_eq!(cfg.thinking_loop.brainstorm_max_tokens, 4096);
}

#[test]
fn thinking_loop_quality_gate_max_tokens_has_fixed_default() {
    let cfg = cfg_with_model_max_tokens(4096);
    assert_eq!(cfg.thinking_loop.quality_gate_max_tokens, 64);
}

#[test]
fn thinking_loop_synthesis_tournament_max_round_tokens_cascades_from_model() {
    let cfg = cfg_with_model_max_tokens(4096);
    assert_eq!(cfg.thinking_loop.synthesis_tournament_max_round_tokens, 4096);
}

#[test]
fn srani_researcher_max_tokens_cascades_from_model() {
    let cfg = cfg_with_model_max_tokens(4096);
    assert_eq!(cfg.srani.researcher_max_tokens, 4096);
}

#[test]
fn srani_distill_max_tokens_cascades_from_model() {
    let cfg = cfg_with_model_max_tokens(4096);
    assert_eq!(cfg.srani.distill_max_tokens, 4096);
}

#[test]
fn srani_gap_synthesis_max_tokens_cascades_from_model() {
    let cfg = cfg_with_model_max_tokens(4096);
    assert_eq!(cfg.srani.gap_synthesis_max_tokens, 4096);
}

#[test]
fn gap_k1_repair_max_tokens_cascades_from_model() {
    let cfg = cfg_with_model_max_tokens(4096);
    assert_eq!(cfg.gap_k1.repair_max_tokens, 4096);
}

#[test]
fn opro_max_tokens_cascades_from_model() {
    let cfg = cfg_with_model_max_tokens(4096);
    assert_eq!(cfg.opro.max_tokens, 4096);
}

#[test]
fn hallucination_check_max_tokens_cascades_from_model() {
    let cfg = cfg_with_model_max_tokens(4096);
    assert_eq!(cfg.hallucination_check_max_tokens, 4096);
}

#[test]
fn generation_search_max_tokens_cascades_from_model() {
    let cfg = cfg_with_model_max_tokens(4096);
    assert_eq!(cfg.generation_search_max_tokens, 4096);
}

#[test]
fn planner_decompose_max_tokens_cascades_from_model() {
    let cfg = cfg_with_model_max_tokens(4096);
    assert_eq!(cfg.planner_decompose_max_tokens, 4096);
}

#[test]
fn planner_review_max_tokens_cascades_from_model() {
    let cfg = cfg_with_model_max_tokens(4096);
    assert_eq!(cfg.planner_review_max_tokens, 4096);
}

#[test]
fn toml_override_takes_precedence_over_model_cascade() {
    let mut tmp = tempfile::NamedTempFile::with_suffix(".toml").unwrap();
    writeln!(tmp, "model_max_tokens = 4096").unwrap();
    writeln!(tmp, "[thinking_loop]").unwrap();
    writeln!(tmp, "archetype_select_max_tokens = 8192").unwrap();
    let cfg = H2AIConfig::load_layered(Some(tmp.path())).expect("valid");
    assert_eq!(cfg.thinking_loop.archetype_select_max_tokens, 8192);
    assert_eq!(cfg.thinking_loop.brainstorm_max_tokens, 4096);
}

#[test]
fn default_config_has_expected_fallback_values() {
    let cfg = H2AIConfig::default();
    assert_eq!(cfg.thinking_loop.archetype_select_max_tokens, 32_768);
    assert_eq!(cfg.thinking_loop.brainstorm_max_tokens, 32_768);
    assert_eq!(cfg.thinking_loop.quality_gate_max_tokens, 64);
    assert_eq!(cfg.thinking_loop.synthesis_tournament_max_round_tokens, 32_768);
    assert_eq!(cfg.srani.researcher_max_tokens, 32_768);
    assert_eq!(cfg.srani.distill_max_tokens, 32_768);
    assert_eq!(cfg.srani.gap_synthesis_max_tokens, 32_768);
    assert_eq!(cfg.gap_k1.repair_max_tokens, 32_768);
    assert_eq!(cfg.opro.max_tokens, 32_768);
    assert_eq!(cfg.hallucination_check_max_tokens, 32_768);
    assert_eq!(cfg.generation_search_max_tokens, 32_768);
    assert_eq!(cfg.planner_decompose_max_tokens, 32_768);
    assert_eq!(cfg.planner_review_max_tokens, 32_768);
}
