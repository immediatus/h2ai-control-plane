use h2ai_config::H2AIConfig;
use rand::Rng;
use rand_distr::{Beta, Distribution};
use std::collections::BTreeMap;

/// Per-arm Beta distribution state for Thompson Sampling.
#[derive(Debug, Clone)]
pub struct BanditArm {
    /// Beta distribution α (successes + prior).
    pub alpha: f64,
    /// Beta distribution β (failures + prior).
    pub beta: f64,
}

impl BanditArm {
    /// Posterior mean: E[θ] = α / (α + β).
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Draw one Beta(α, β) sample via Thompson Sampling.
    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        // alpha and beta are always ≥ 1.0 by construction; Beta::new never panics here.
        let dist = Beta::new(self.alpha, self.beta).expect("alpha/beta >= 1.0");
        dist.sample(rng)
    }
}

/// Thompson Sampling bandit state for N (agent count) selection.
///
/// Arms are keyed by N value (1..=min(6, N_max_USL)).
/// Phased activation:
/// - Phase 0 (k_tasks < cfg.bandit_phase0_k): return max arm (N_max_USL); no update.
/// - Phase 1 (phase0_k ≤ k_tasks < phase1_k): ε-greedy TS.
/// - Phase 2 (k_tasks ≥ phase1_k): pure Thompson Sampling.
#[derive(Debug, Clone)]
pub struct BanditState {
    pub arms: BTreeMap<u32, BanditArm>,
    /// Total tasks completed (includes Phase 0 observations that weren't used for updates).
    pub k_tasks: u32,
    /// Version hash of the adapter set at initialization.
    /// When it changes, `soft_reset` should be called.
    pub adapter_version_hash: u64,
    /// Snapshot of the initial prior used for soft reset.
    initial_prior: BTreeMap<u32, BanditArm>,
}

impl BanditState {
    /// Create a new BanditState with a warm prior centered on `n_max_usl`.
    pub fn new(n_max_usl: u32, adapter_version_hash: u64) -> Self {
        let arm_keys: Vec<u32> = (1..=n_max_usl.max(1).min(6)).collect();
        let arms = warm_prior(n_max_usl, &arm_keys);
        let initial_prior = warm_prior(n_max_usl, &arm_keys);
        Self {
            arms,
            k_tasks: 0,
            adapter_version_hash,
            initial_prior,
        }
    }

    /// Select an arm (N value) according to the current phase.
    pub fn select(&self, cfg: &H2AIConfig) -> u32 {
        let n_max_arm = *self.arms.keys().last().unwrap_or(&1);

        // Phase 0: no bandit, return N_max
        if self.k_tasks < cfg.bandit_phase0_k {
            return n_max_arm;
        }

        let mut rng = rand::thread_rng();

        // Phase 1: ε-greedy
        if self.k_tasks < cfg.bandit_phase1_k {
            if rng.gen::<f64>() < cfg.bandit_epsilon {
                let idx = rng.gen_range(0..self.arms.len());
                return *self.arms.keys().nth(idx).unwrap_or(&n_max_arm);
            }
        }

        // Phase 2 (and TS part of Phase 1): pure Thompson Sampling
        self.arms
            .iter()
            .map(|(&n, arm)| (n, arm.sample(&mut rng)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(n, _)| n)
            .unwrap_or(n_max_arm)
    }

    /// Update the posterior after a task completes.
    ///
    /// Reward priority: Tier 1 (oracle) overrides Tier 3 (LLM judge) when both present.
    /// Tier 3 soft update: alpha += score, beta += (1 − score).
    pub fn update(&mut self, n_used: u32, tier1_passed: Option<bool>, tier3_score: Option<f64>) {
        let arm = match self.arms.get_mut(&n_used) {
            Some(a) => a,
            None => return,
        };
        if let Some(passed) = tier1_passed {
            // Tier 1 oracle is ground truth — hard binary update.
            if passed {
                arm.alpha += 1.0;
            } else {
                arm.beta += 1.0;
            }
        } else if let Some(score) = tier3_score {
            // Tier 3: partial credit for graded scores.
            let s = score.clamp(0.0, 1.0);
            arm.alpha += s;
            arm.beta += 1.0 - s;
        }
        self.k_tasks += 1;
    }

    /// Apply a soft reset toward the initial prior on adapter version change.
    ///
    /// `decay` controls how much of the initial prior is injected:
    /// - new_param = current_param × (1 − decay) + prior_param × decay
    /// - Default decay = 0.3 (preserves 70% of learned posterior).
    pub fn soft_reset(&mut self, decay: f64) {
        let decay = decay.clamp(0.0, 1.0);
        for (n, arm) in &mut self.arms {
            if let Some(prior) = self.initial_prior.get(n) {
                arm.alpha = arm.alpha * (1.0 - decay) + prior.alpha * decay;
                arm.beta = arm.beta * (1.0 - decay) + prior.beta * decay;
            }
        }
        self.k_tasks = 0;
    }

    /// Apply the SelfOptimizer suggestion (Decision 4).
    ///
    /// When suggested N < current N AND the current arm is near-optimal (mean > 0.9),
    /// add a weak pull toward the suggested arm's alpha (+0.5) without changing others.
    pub fn apply_optimizer_hint(&mut self, current_n: u32, suggested_n: u32) {
        if suggested_n >= current_n {
            return; // Only act when suggestion is to reduce N
        }
        let current_mean = self.arms.get(&current_n).map(|a| a.mean()).unwrap_or(0.0);
        if current_mean > 0.9 {
            if let Some(arm) = self.arms.get_mut(&suggested_n) {
                arm.alpha += 0.5;
            }
        }
    }
}

/// Build a warm prior `BTreeMap<N, BanditArm>` with a Gaussian weight centered on `n_max_usl`.
///
/// σ=2: arms within 2 of N_max_USL get meaningful prior weight;
/// arms far from N_max_USL start near Beta(1, 6) — weighted against but not excluded.
pub fn warm_prior(n_max_usl: u32, arm_keys: &[u32]) -> BTreeMap<u32, BanditArm> {
    let sigma_sq = 4.0_f64; // σ=2 → σ²=4
    let prior_strength = 5.0_f64;
    arm_keys
        .iter()
        .map(|&n| {
            let dist = (n as f64 - n_max_usl as f64).abs();
            let weight = (-dist.powi(2) / (2.0 * sigma_sq)).exp();
            (
                n,
                BanditArm {
                    alpha: prior_strength * weight + 1.0,
                    beta: prior_strength * (1.0 - weight) + 1.0,
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> H2AIConfig {
        H2AIConfig::default()
    }

    // ── warm_prior ────────────────────────────────────────────────────────────

    #[test]
    fn warm_prior_arm_at_n_max_has_highest_mean() {
        let n_max = 4u32;
        let keys: Vec<u32> = (1..=6).collect();
        let prior = warm_prior(n_max, &keys);
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
        // N=1 when N_max=6: distance=5 → weight ≈ exp(-25/8) ≈ 0.044 → alpha ≈ 1.22, beta ≈ 5.78
        let prior = warm_prior(6, &[1u32, 6]);
        let mean_1 = prior[&1].mean();
        let mean_6 = prior[&6].mean();
        assert!(
            mean_6 > mean_1,
            "N=6 (at N_max) must have higher mean than N=1; got {mean_6:.3} vs {mean_1:.3}"
        );
    }

    // ── phase transitions ─────────────────────────────────────────────────────

    #[test]
    fn select_phase0_returns_n_max_arm() {
        let state = BanditState::new(4, 0);
        let cfg = cfg();
        // k_tasks = 0 < phase0_k = 10 → Phase 0
        for _ in 0..20 {
            assert_eq!(state.select(&cfg), 4, "Phase 0 must return N_max arm=4");
        }
    }

    #[test]
    fn update_increments_k_tasks() {
        let mut state = BanditState::new(4, 0);
        assert_eq!(state.k_tasks, 0);
        state.update(4, None, Some(0.8));
        assert_eq!(state.k_tasks, 1);
    }

    #[test]
    fn phase1_activates_at_phase0_k() {
        let mut state = BanditState::new(4, 0);
        let cfg = cfg();
        // Advance to k_tasks = phase0_k = 10 (just entering Phase 1)
        for _ in 0..10 {
            state.update(4, None, Some(0.8));
        }
        assert_eq!(state.k_tasks, 10);
        // Phase 1: should NOT always return N_max (ε-greedy introduces variance)
        // but we can at least check it doesn't panic and returns a valid arm
        let n = state.select(&cfg);
        assert!(
            state.arms.contains_key(&n),
            "selected N must be a valid arm"
        );
    }

    // ── reward updates ────────────────────────────────────────────────────────

    #[test]
    fn tier1_pass_increments_alpha() {
        let mut state = BanditState::new(4, 0);
        let alpha_before = state.arms[&4].alpha;
        state.update(4, Some(true), None);
        assert!(
            (state.arms[&4].alpha - (alpha_before + 1.0)).abs() < 1e-9,
            "Tier 1 pass must increment alpha by 1"
        );
        assert!(
            (state.arms[&4].beta - warm_prior(4, &[4])[&4].beta).abs() < 1e-9,
            "Tier 1 pass must not change beta"
        );
    }

    #[test]
    fn tier1_fail_increments_beta() {
        let mut state = BanditState::new(4, 0);
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
        let mut state = BanditState::new(4, 0);
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
        let mut state_t1 = BanditState::new(4, 0);
        let mut state_t3 = BanditState::new(4, 0);

        // Update with Tier 1 pass + Tier 3 score=0.5
        state_t1.update(4, Some(true), Some(0.5));
        state_t3.update(4, None, Some(0.5));

        let alpha_t1 = state_t1.arms[&4].alpha;
        let alpha_t3 = state_t3.arms[&4].alpha;

        // Tier 1 adds 1.0; Tier 3 adds 0.5 — if Tier 1 overrides, result should match pure Tier 1
        assert!(
            (alpha_t1 - alpha_t3).abs() > 0.3,
            "Tier 1 must produce different update than Tier 3; diff={:.3}",
            (alpha_t1 - alpha_t3).abs()
        );
        // More precisely: Tier 1 adds exactly 1.0, Tier 3 would add 0.5
        let prior_alpha = warm_prior(4, &[4])[&4].alpha;
        assert!(
            (alpha_t1 - (prior_alpha + 1.0)).abs() < 1e-9,
            "Tier 1 must add exactly 1.0 to alpha"
        );
    }

    // ── soft_reset ────────────────────────────────────────────────────────────

    #[test]
    fn soft_reset_decay_0_3_preserves_70_percent() {
        let mut state = BanditState::new(4, 0);
        // Train heavily on arm 4: add 100 successes
        for _ in 0..100 {
            state.update(4, Some(true), None);
        }
        let alpha_before = state.arms[&4].alpha;
        let prior_alpha = state.initial_prior[&4].alpha;

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
        let mut state = BanditState::new(4, 0);
        for _ in 0..50 {
            state.update(4, Some(true), None);
        }
        state.soft_reset(1.0);

        let prior = warm_prior(4, &(1u32..=4).collect::<Vec<_>>());
        for (&n, arm) in &state.arms {
            let p = &prior[&n];
            assert!(
                (arm.alpha - p.alpha).abs() < 1e-9 && (arm.beta - p.beta).abs() < 1e-9,
                "decay=1.0 must restore to exact prior for arm N={n}"
            );
        }
    }

    // ── optimizer hint ───────────────────────────────────────────────────────

    #[test]
    fn optimizer_hint_nudges_lower_arm_when_current_is_near_optimal() {
        let mut state = BanditState::new(4, 0);
        // Force arm 4 to near-optimal mean by adding many successes
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
        let mut state = BanditState::new(4, 0);
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
}
