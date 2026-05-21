use h2ai_config::H2AIConfig;
use rand::Rng;
use rand_distr::{Beta, Distribution};
use std::collections::BTreeMap;

/// Per-arm Beta distribution state for Thompson Sampling.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BanditArm {
    /// Beta distribution α (successes + prior).
    pub alpha: f64,
    /// Beta distribution β (failures + prior).
    pub beta: f64,
}

impl BanditArm {
    /// Posterior mean: E[θ] = α / (α + β).
    #[must_use]
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
/// Arms are keyed by N value (`1..=min(bandit_n_max_arms`, `N_max_USL`)).
/// Phased activation:
/// - Phase 0 (`k_tasks` < `cfg.bandit_phase0_k)`: return max arm (`N_max_USL`); no update.
/// - Phase 1 (`phase0_k` ≤ `k_tasks` < `phase1_k)`: ε-greedy TS.
/// - Phase 2 (`k_tasks` ≥ `phase1_k)`: pure Thompson Sampling.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    /// Create a new `BanditState` with a warm prior centered on `n_max_usl`.
    #[must_use]
    pub fn new(
        n_max_usl: u32,
        adapter_version_hash: u64,
        max_arms: u32,
        prior_sigma: f64,
        prior_strength: f64,
    ) -> Self {
        let arm_keys: Vec<u32> = (1..=n_max_usl.clamp(1, max_arms)).collect();
        let arms = warm_prior(n_max_usl, &arm_keys, prior_sigma, prior_strength);
        let initial_prior = warm_prior(n_max_usl, &arm_keys, prior_sigma, prior_strength);
        Self {
            arms,
            k_tasks: 0,
            adapter_version_hash,
            initial_prior,
        }
    }

    /// Select an arm (N value) according to the current phase.
    #[must_use]
    pub fn select(&self, cfg: &H2AIConfig) -> u32 {
        let n_max_arm = *self.arms.keys().last().unwrap_or(&1);

        // Phase 0: no bandit, return N_max
        if self.k_tasks < cfg.bandit_phase0_k {
            return n_max_arm;
        }

        let mut rng = rand::thread_rng();

        // Phase 1: ε-greedy
        if self.k_tasks < cfg.bandit_phase1_k && rng.gen::<f64>() < cfg.bandit_epsilon {
            let idx = rng.gen_range(0..self.arms.len());
            return *self.arms.keys().nth(idx).unwrap_or(&n_max_arm);
        }

        // Phase 2 (and TS part of Phase 1): pure Thompson Sampling
        self.arms
            .iter()
            .map(|(&n, arm)| (n, arm.sample(&mut rng)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(n_max_arm, |(n, _)| n)
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
    /// - `new_param` = `current_param` × (1 − decay) + `prior_param` × decay
    /// - Default decay = 0.3 (preserves 70% of learned posterior).
    pub fn soft_reset(&mut self, decay: f64) {
        let decay = decay.clamp(0.0, 1.0);
        for (n, arm) in &mut self.arms {
            if let Some(prior) = self.initial_prior.get(n) {
                arm.alpha = arm.alpha.mul_add(1.0 - decay, prior.alpha * decay);
                arm.beta = arm.beta.mul_add(1.0 - decay, prior.beta * decay);
            }
        }
        self.k_tasks = 0;
    }

    /// Apply the `SelfOptimizer` suggestion (Decision 4).
    ///
    /// When suggested N < current N AND the current arm is near-optimal (mean > 0.9),
    /// add a weak pull toward the suggested arm's alpha (+0.5) without changing others.
    pub fn apply_optimizer_hint(&mut self, current_n: u32, suggested_n: u32) {
        if suggested_n >= current_n {
            return; // Only act when suggestion is to reduce N
        }
        let current_mean = self.arms.get(&current_n).map_or(0.0, BanditArm::mean);
        if current_mean > 0.9 {
            if let Some(arm) = self.arms.get_mut(&suggested_n) {
                arm.alpha += 0.5;
            }
        }
    }
}

/// Build a warm prior `BTreeMap<N, BanditArm>` with a Gaussian weight centered on `n_max_usl`.
///
/// `prior_sigma` controls width: arms within `prior_sigma` of `N_max_USL` get meaningful weight;
/// arms far from `N_max_USL` start near Beta(1, `prior_strength+1`) — weighted against but not excluded.
#[must_use]
pub fn warm_prior(
    n_max_usl: u32,
    arm_keys: &[u32],
    prior_sigma: f64,
    prior_strength: f64,
) -> BTreeMap<u32, BanditArm> {
    let sigma_sq = prior_sigma * prior_sigma;
    arm_keys
        .iter()
        .map(|&n| {
            let dist = (f64::from(n) - f64::from(n_max_usl)).abs();
            let weight = (-dist.powi(2) / (2.0 * sigma_sq)).exp();
            (
                n,
                BanditArm {
                    alpha: prior_strength.mul_add(weight, 1.0),
                    beta: prior_strength.mul_add(1.0 - weight, 1.0),
                },
            )
        })
        .collect()
}
