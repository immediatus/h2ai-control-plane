#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    clippy::unused_async,
    clippy::default_trait_access,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::similar_names,
    clippy::match_same_arms,
    clippy::literal_string_with_formatting_args,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::useless_format,
    clippy::option_if_let_else,
    clippy::map_unwrap_or,
    clippy::cloned_instead_of_copied,
    clippy::trivially_copy_pass_by_ref,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::needless_pass_by_value,
    clippy::explicit_iter_loop,
    clippy::needless_borrow,
    clippy::large_futures,
    clippy::manual_string_new,
    clippy::needless_lifetimes,
    clippy::elidable_lifetime_names,
    clippy::redundant_else,
    clippy::stable_sort_primitive,
    clippy::type_complexity,
    clippy::wildcard_imports,
    clippy::single_match_else,
    clippy::missing_fields_in_debug,
    clippy::doc_link_with_quotes,
    clippy::implicit_hasher,
    clippy::needless_collect,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::needless_type_cast,
    clippy::unreadable_literal,
    clippy::no_effect_underscore_binding
)]
use h2ai_api::opro::{
    check_graduation, compute_ema, extract_template_variables, should_trigger_opro,
    thompson_sample, validate_opro_response,
};
use h2ai_config::OproConfig;
use h2ai_types::prompt_variant::PromptBanditArm;

fn default_opro_cfg() -> OproConfig {
    OproConfig::default()
}

#[test]
fn compute_ema_basic() {
    // window=1 → alpha=1.0 → new_value completely replaces old_ema
    let result = compute_ema(0.5, 0.8, 1);
    assert!(
        (result - 0.8).abs() < 1e-9,
        "window=1 should give alpha=1.0"
    );

    // window=9 → alpha = 2/(9+1) = 0.2
    let result2 = compute_ema(0.5, 1.0, 9);
    let expected = 0.2f64.mul_add(1.0, 0.8 * 0.5);
    assert!((result2 - expected).abs() < 1e-9);
}

#[test]
fn extract_template_variables_finds_vars() {
    let tmpl = "Hello {name}, your score is {score} out of {total}.";
    let mut vars = extract_template_variables(tmpl);
    vars.sort();
    assert_eq!(vars, vec!["name", "score", "total"]);
}

#[test]
fn extract_template_variables_no_vars() {
    let vars = extract_template_variables("No placeholders here.");
    assert!(vars.is_empty());
}

#[test]
fn validate_opro_response_ok() {
    let original = "Task: {task}, context: {context}";
    let candidate = "Improved task: {task}. Use context: {context} wisely.";
    assert!(validate_opro_response(original, candidate).is_ok());
}

#[test]
fn validate_opro_response_missing_var() {
    let original = "Task: {task}, context: {context}";
    let candidate = "Improved task: {task}."; // missing {context}
    let err = validate_opro_response(original, candidate).unwrap_err();
    assert!(err.contains(&"context".to_string()));
}

#[test]
fn should_trigger_opro_below_threshold() {
    let mut cfg = default_opro_cfg();
    cfg.enabled = true;
    cfg.trigger_j_eff_threshold = 0.6;
    cfg.min_tasks_before_trigger = 10;

    // j_eff=0.5 < 0.6 and 15 >= 10 → should trigger
    assert!(should_trigger_opro(0.5, 15, 0, &cfg));
}

#[test]
fn should_trigger_opro_not_enough_tasks() {
    let mut cfg = default_opro_cfg();
    cfg.enabled = true;
    cfg.trigger_j_eff_threshold = 0.6;
    cfg.min_tasks_before_trigger = 10;

    // Only 5 tasks — below min_tasks_before_trigger → should NOT trigger
    assert!(!should_trigger_opro(0.5, 5, 0, &cfg));
}

#[test]
fn should_trigger_opro_disabled() {
    let mut cfg = default_opro_cfg();
    cfg.enabled = false;
    cfg.trigger_j_eff_threshold = 0.6;
    cfg.min_tasks_before_trigger = 1;

    assert!(!should_trigger_opro(0.1, 100, 0, &cfg));
}

#[test]
fn should_trigger_opro_suppressed() {
    let mut cfg = default_opro_cfg();
    cfg.enabled = true;
    cfg.trigger_j_eff_threshold = 0.6;
    cfg.min_tasks_before_trigger = 5;

    // suppress_until_n_tasks = 20, n_tasks_total = 15 → suppressed
    assert!(!should_trigger_opro(0.3, 15, 20, &cfg));
}

#[test]
fn thompson_sample_picks_best_arm() {
    let arms = vec![
        PromptBanditArm {
            variant_id: "a".to_string(),
            alpha: 1.0,
            beta: 9.0, // mean = 0.1
        },
        PromptBanditArm {
            variant_id: "b".to_string(),
            alpha: 8.0,
            beta: 2.0, // mean = 0.8
        },
        PromptBanditArm {
            variant_id: "c".to_string(),
            alpha: 5.0,
            beta: 5.0, // mean = 0.5
        },
    ];
    let selected = thompson_sample(&arms);
    assert_eq!(selected, Some("b"), "should pick arm with highest mean");
}

#[test]
fn thompson_sample_empty() {
    let arms: Vec<PromptBanditArm> = vec![];
    assert_eq!(thompson_sample(&arms), None);
}

#[test]
fn check_graduation_promotes_when_above_margin() {
    let mut cfg = default_opro_cfg();
    cfg.graduation_tasks = 20;
    cfg.promotion_margin = 0.05;

    let arms = vec![
        PromptBanditArm {
            variant_id: "seed".to_string(),
            alpha: 5.0,
            beta: 5.0, // mean = 0.5
        },
        PromptBanditArm {
            variant_id: "candidate".to_string(),
            alpha: 8.0,
            beta: 2.0, // mean = 0.8 (> 0.5 + 0.05)
        },
    ];
    assert!(check_graduation("candidate", &arms, 25, &cfg));
}

#[test]
fn check_graduation_not_enough_tasks() {
    let cfg = default_opro_cfg();
    let arms = vec![
        PromptBanditArm {
            variant_id: "seed".to_string(),
            alpha: 1.0,
            beta: 9.0,
        },
        PromptBanditArm {
            variant_id: "v2".to_string(),
            alpha: 9.0,
            beta: 1.0,
        },
    ];
    // n_tasks_total = 5 < graduation_tasks = 20 → false
    assert!(!check_graduation("v2", &arms, 5, &cfg));
}
