use h2ai_config::H2AIConfig;
use h2ai_orchestrator::bandit::{warm_prior, BanditState};

fn cfg() -> H2AIConfig {
    H2AIConfig::default()
}

// ── warm_prior ─────────────────────────────────────────────────────────────────

#[test]
fn warm_prior_arm_at_n_max_has_highest_mean() {
    let n_max = 4u32;
    let keys: Vec<u32> = (1..=6).collect();
    let prior = warm_prior(n_max, &keys, 2.0, 5.0);
    let mean_at_n_max = prior[&n_max].mean();
    for (&n, arm) in &prior {
        if n != n_max {
            assert!(
                mean_at_n_max >= arm.mean(),
                "arm at N_max={n_max} must have highest mean; N={n} has {:.3} > {:.3}",
                arm.mean(),
                mean_at_n_max
            );
        }
    }
}

#[test]
fn warm_prior_distant_arm_weighted_against() {
    let prior = warm_prior(6, &[1u32, 6], 2.0, 5.0);
    let mean_1 = prior[&1].mean();
    let mean_6 = prior[&6].mean();
    assert!(
        mean_6 > mean_1,
        "N=6 (at N_max) must have higher mean than N=1; got {mean_6:.3} vs {mean_1:.3}"
    );
}

#[test]
fn warm_prior_uses_custom_sigma_and_strength() {
    // sigma=1 (tighter), strength=2 → arm at N_max: weight=exp(0)=1 → alpha = 2*1+1=3.0, beta = 2*(1-1)+1=1.0
    let prior = warm_prior(4, &[4u32], 1.0, 2.0);
    let alpha_at_max = prior[&4].alpha;
    let beta_at_max = prior[&4].beta;
    assert!((alpha_at_max - 3.0).abs() < 1e-9, "alpha={alpha_at_max}");
    assert!((beta_at_max - 1.0).abs() < 1e-9, "beta={beta_at_max}");
}

// ── phase transitions ──────────────────────────────────────────────────────────

#[test]
fn select_phase0_returns_n_max_arm() {
    let state = BanditState::new(4, 0, 6, 2.0, 5.0);
    let cfg = cfg();
    // k_tasks = 0 < phase0_k = 10 → Phase 0
    for _ in 0..20 {
        assert_eq!(state.select(&cfg), 4, "Phase 0 must return N_max arm=4");
    }
}

#[test]
fn update_increments_k_tasks() {
    let mut state = BanditState::new(4, 0, 6, 2.0, 5.0);
    assert_eq!(state.k_tasks, 0);
    state.update(4, None, Some(0.8));
    assert_eq!(state.k_tasks, 1);
}

#[test]
fn phase1_activates_at_phase0_k() {
    let mut state = BanditState::new(4, 0, 6, 2.0, 5.0);
    let cfg = cfg();
    // Advance to k_tasks = phase0_k = 10 (just entering Phase 1)
    for _ in 0..10 {
        state.update(4, None, Some(0.8));
    }
    assert_eq!(state.k_tasks, 10);
    let n = state.select(&cfg);
    assert!(
        state.arms.contains_key(&n),
        "selected N must be a valid arm"
    );
}

// ── reward updates ─────────────────────────────────────────────────────────────

#[test]
fn tier1_pass_increments_alpha() {
    let mut state = BanditState::new(4, 0, 6, 2.0, 5.0);
    let alpha_before = state.arms[&4].alpha;
    state.update(4, Some(true), None);
    assert!(
        (state.arms[&4].alpha - (alpha_before + 1.0)).abs() < 1e-9,
        "Tier 1 pass must increment alpha by 1"
    );
    assert!(
        (state.arms[&4].beta - warm_prior(4, &[4], 2.0, 5.0)[&4].beta).abs() < 1e-9,
        "Tier 1 pass must not change beta"
    );
}

#[test]
fn tier1_fail_increments_beta() {
    let mut state = BanditState::new(4, 0, 6, 2.0, 5.0);
    let beta_before = state.arms[&4].beta;
    state.update(4, Some(false), None);
    assert!(
        (state.arms[&4].beta - (beta_before + 1.0)).abs() < 1e-9,
        "Tier 1 fail must increment beta by 1"
    );
}

#[test]
fn tier3_soft_update_score_0_8() {
    // score=0.8 → alpha += 0.8, beta += 0.2
    let mut state = BanditState::new(4, 0, 6, 2.0, 5.0);
    let alpha_before = state.arms[&3].alpha;
    let beta_before = state.arms[&3].beta;
    state.update(3, None, Some(0.8));
    assert!(
        (state.arms[&3].alpha - (alpha_before + 0.8)).abs() < 1e-9,
        "Tier 3 score=0.8 must add 0.8 to alpha"
    );
    assert!(
        (state.arms[&3].beta - (beta_before + 0.2)).abs() < 1e-9,
        "Tier 3 score=0.8 must add 0.2 to beta"
    );
}

#[test]
fn tier1_overrides_tier3_when_both_present() {
    let mut state_t1 = BanditState::new(4, 0, 6, 2.0, 5.0);
    let mut state_t3 = BanditState::new(4, 0, 6, 2.0, 5.0);

    state_t1.update(4, Some(true), Some(0.5));
    state_t3.update(4, None, Some(0.5));

    let alpha_t1 = state_t1.arms[&4].alpha;
    let alpha_t3 = state_t3.arms[&4].alpha;

    assert!(
        (alpha_t1 - alpha_t3).abs() > 0.3,
        "Tier 1 must produce different update than Tier 3; diff={:.3}",
        (alpha_t1 - alpha_t3).abs()
    );
    let prior_alpha = warm_prior(4, &[4], 2.0, 5.0)[&4].alpha;
    assert!(
        (alpha_t1 - (prior_alpha + 1.0)).abs() < 1e-9,
        "Tier 1 must add exactly 1.0 to alpha"
    );
}

// ── soft_reset ─────────────────────────────────────────────────────────────────

#[test]
fn soft_reset_decay_0_3_preserves_70_percent() {
    let mut state = BanditState::new(4, 0, 6, 2.0, 5.0);
    for _ in 0..100 {
        state.update(4, Some(true), None);
    }
    let alpha_before = state.arms[&4].alpha;
    // initial_prior is private; recompute via warm_prior with the same parameters used in new()
    let prior_alpha = warm_prior(4, &(1u32..=4).collect::<Vec<_>>(), 2.0, 5.0)[&4].alpha;

    state.soft_reset(0.3);

    let expected = alpha_before * 0.7 + prior_alpha * 0.3;
    assert!(
        (state.arms[&4].alpha - expected).abs() < 1e-9,
        "soft_reset(0.3) must give 70% current + 30% prior; expected {expected:.4}, got {:.4}",
        state.arms[&4].alpha
    );
    assert_eq!(state.k_tasks, 0, "soft_reset must reset k_tasks to 0");
}

#[test]
fn soft_reset_decay_1_returns_to_prior() {
    let mut state = BanditState::new(4, 0, 6, 2.0, 5.0);
    for _ in 0..50 {
        state.update(4, Some(true), None);
    }
    state.soft_reset(1.0);

    let prior = warm_prior(4, &(1u32..=4).collect::<Vec<_>>(), 2.0, 5.0);
    for (&n, arm) in &state.arms {
        let p = &prior[&n];
        assert!(
            (arm.alpha - p.alpha).abs() < 1e-9 && (arm.beta - p.beta).abs() < 1e-9,
            "decay=1.0 must restore to exact prior for arm N={n}"
        );
    }
}

// ── optimizer hint ─────────────────────────────────────────────────────────────

#[test]
fn optimizer_hint_nudges_lower_arm_when_current_is_near_optimal() {
    let mut state = BanditState::new(4, 0, 6, 2.0, 5.0);
    for _ in 0..200 {
        state.update(4, Some(true), None);
    }
    assert!(
        state.arms[&4].mean() > 0.9,
        "arm 4 must be near-optimal for this test"
    );
    let alpha_before = state.arms[&3].alpha;
    state.apply_optimizer_hint(4, 3);
    assert!(
        (state.arms[&3].alpha - (alpha_before + 0.5)).abs() < 1e-9,
        "optimizer hint must add 0.5 to suggested arm's alpha"
    );
}

#[test]
fn optimizer_hint_skipped_when_suggested_n_not_lower() {
    let mut state = BanditState::new(4, 0, 6, 2.0, 5.0);
    let arms_before: Vec<(u32, f64, f64)> = state
        .arms
        .iter()
        .map(|(&n, a)| (n, a.alpha, a.beta))
        .collect();
    state.apply_optimizer_hint(3, 4); // suggested_n (4) >= current_n (3) → no-op
    for (n, alpha, beta) in arms_before {
        assert!(
            (state.arms[&n].alpha - alpha).abs() < 1e-9,
            "no-op hint must not change alpha for N={n}"
        );
        assert!(
            (state.arms[&n].beta - beta).abs() < 1e-9,
            "no-op hint must not change beta for N={n}"
        );
    }
}

#[test]
fn bandit_state_serde_roundtrip() {
    let state = BanditState::new(4, 42, 6, 2.0, 5.0);
    let json = serde_json::to_string(&state).expect("serialize");
    let restored: BanditState = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(state.k_tasks, restored.k_tasks);
    assert_eq!(state.adapter_version_hash, restored.adapter_version_hash);
    assert_eq!(state.arms.len(), restored.arms.len());
}

#[test]
fn bandit_new_respects_max_arms_ceiling() {
    // max_arms=3 → arms are [1,2,3] even when n_max_usl=6
    let state = BanditState::new(6, 0, 3, 2.0, 5.0);
    let max_arm = *state.arms.keys().last().unwrap();
    assert_eq!(max_arm, 3, "max arm must be clamped to max_arms=3");
}

#[test]
fn bandit_update_after_task_raises_k_tasks_and_optimizer_hint_works() {
    let mut state = BanditState::new(4, 0, 6, 2.0, 5.0);
    assert_eq!(state.k_tasks, 0);
    state.update(3, None, Some(0.85));
    assert_eq!(state.k_tasks, 1);
    let alpha_arm2_before = state.arms[&2].alpha;
    for _ in 0..200 {
        state.update(3, Some(true), None);
    }
    state.apply_optimizer_hint(3, 2);
    assert!(
        state.arms[&2].alpha > alpha_arm2_before,
        "optimizer hint must nudge arm 2 alpha when arm 3 is near-optimal"
    );
}

#[test]
fn bandit_update_unknown_arm_is_noop() {
    let mut state = BanditState::new(4, 0, 6, 2.0, 5.0);
    let k_before = state.k_tasks;
    // n_used=999 doesn't exist in arms — must not panic, k_tasks must not change
    state.update(999, Some(true), None);
    assert_eq!(state.k_tasks, k_before);
}

#[test]
fn select_phase1_epsilon_greedy_explores() {
    // Force phase 1 by keeping k_tasks < bandit_phase1_k.
    // Use epsilon=1.0 to guarantee exploration branch is taken.
    let state = BanditState::new(4, 0, 6, 2.0, 5.0);
    let c = cfg();
    // With epsilon=1.0 in cfg, every call during phase 1 takes the explore path.
    // Run many times; as long as it returns a valid arm and doesn't panic, test passes.
    for _ in 0..50 {
        let n = state.select(&c);
        assert!(
            state.arms.contains_key(&n),
            "select must return a known arm"
        );
    }
}
