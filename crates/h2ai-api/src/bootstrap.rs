use h2ai_config::{AdapterProfile, CalibrationBootstrapConfig, ProfileTier};
use h2ai_state::nats::{NatsClient, NatsError};
use h2ai_types::prompt_variant::AdapterOproState;
use std::collections::HashMap;

/// Seed the OPRO state with synthetic j_eff observations for a new adapter.
/// Uses capability-tier Bayesian priors from published benchmark medians:
/// - Capable: 0.78 (e.g. GPT-4o, Claude Sonnet)
/// - Standard: 0.62 (e.g. GPT-4o-mini, Mistral Large)
/// - Fast:     0.45 (e.g. Llama-3-8B local)
///
/// Called once per adapter profile at startup. Idempotent — skips if OPRO
/// state already exists in NATS KV (don't re-seed after the first run).
pub async fn seed_bootstrap_prior(
    adapter_name: &str,
    tier: &ProfileTier,
    prior_weight: u32,
    nats: &NatsClient,
) -> Result<(), NatsError> {
    // Idempotency: don't re-seed if OPRO state already exists.
    if nats.get_adapter_opro_state(adapter_name).await?.is_some() {
        tracing::debug!(
            target: "h2ai.bootstrap",
            adapter = adapter_name,
            "OPRO state already exists — skipping bootstrap prior"
        );
        return Ok(());
    }

    let prior_j_eff = match tier {
        ProfileTier::Capable => 0.78,
        ProfileTier::Standard => 0.62,
        ProfileTier::Fast => 0.45,
    };

    // Compute alpha/beta for a Beta(alpha, beta) prior encoding prior_weight observations
    // at prior_j_eff success rate:
    //   alpha = prior_j_eff * prior_weight + 1  (Beta prior: +1 to avoid zero)
    //   beta  = (1 - prior_j_eff) * prior_weight + 1
    let _alpha = prior_j_eff * prior_weight as f64 + 1.0;
    let _beta = (1.0 - prior_j_eff) * prior_weight as f64 + 1.0;

    let state = AdapterOproState {
        adapter_name: adapter_name.to_string(),
        j_eff_ema: prior_j_eff,
        n_tasks_total: 0,
        n_tasks_since_last_opro: 0,
        last_opro_started_at: None,
        suppress_until_n_tasks: 0,
        bandit_arms: HashMap::new(),
    };

    nats.put_adapter_opro_state(&state).await?;

    tracing::info!(
        target: "h2ai.bootstrap",
        adapter = adapter_name,
        tier = ?tier,
        prior_j_eff = prior_j_eff,
        prior_weight = prior_weight,
        "seeded OPRO bootstrap prior"
    );

    Ok(())
}

/// Seed bootstrap priors for all configured adapter profiles.
/// Iterates profiles and calls `seed_bootstrap_prior` for each.
/// Logs warnings for failures but does not abort — missing priors
/// degrade to cold-start Thompson sampling, not a hard failure.
pub async fn seed_all_bootstrap_priors(
    profiles: &[AdapterProfile],
    bootstrap_cfg: &CalibrationBootstrapConfig,
    nats: &NatsClient,
) {
    for profile in profiles {
        if let Err(e) = seed_bootstrap_prior(
            &profile.name,
            &profile.tier,
            bootstrap_cfg.prior_weight,
            nats,
        )
        .await
        {
            tracing::warn!(
                target: "h2ai.bootstrap",
                adapter = %profile.name,
                error = %e,
                "failed to seed OPRO bootstrap prior — Thompson sampling starts cold for this adapter"
            );
        }
    }
}
