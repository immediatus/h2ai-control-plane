//! OPRO — Optimization by Prompt Retrieval.
//!
//! After each task completes, `run_opro_trigger` is called (as a background tokio task)
//! to update the `j_eff` EMA, decide whether to generate an improved prompt variant, and
//! maintain the per-adapter Thompson-sampling bandit state.

use h2ai_config::{H2AIConfig, OproConfig};
use h2ai_state::backend::OproStore;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::prompt_variant::{
    AdapterOproState, PromptBanditArm, PromptVariant, PromptVariantSource,
};
use h2ai_types::sizing::TauValue;

// ── Pure helper functions ─────────────────────────────────────────────────────

/// EMA update: alpha = 2 / (window + 1).
#[must_use]
pub fn compute_ema(old_ema: f64, new_value: f64, window: u32) -> f64 {
    let alpha = 2.0 / (f64::from(window) + 1.0);
    alpha * new_value + (1.0 - alpha) * old_ema
}

/// Extract `{variable}` names from a prompt template string.
#[must_use]
pub fn extract_template_variables(template: &str) -> Vec<String> {
    let re = regex::Regex::new(r"\{(\w+)\}").expect("valid regex");
    re.captures_iter(template)
        .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
        .collect()
}

/// Validate that all variables from the original template are still present in the new variant.
///
/// Returns `Ok(())` if valid, `Err(missing_vars)` if any are absent.
pub fn validate_opro_response(original: &str, candidate: &str) -> Result<(), Vec<String>> {
    let required = extract_template_variables(original);
    let present = extract_template_variables(candidate);
    let missing: Vec<String> = required
        .into_iter()
        .filter(|v| !present.contains(v))
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(missing)
    }
}

/// Decide whether to trigger OPRO.
#[must_use]
pub fn should_trigger_opro(
    j_eff_ema: f64,
    n_tasks_total: u32,
    suppress_until_n_tasks: u32,
    cfg: &OproConfig,
) -> bool {
    cfg.enabled
        && j_eff_ema < cfg.trigger_j_eff_threshold
        && n_tasks_total >= cfg.min_tasks_before_trigger
        && n_tasks_total >= suppress_until_n_tasks
}

/// Thompson sample: draw from Beta(alpha, beta) using the posterior mean as a proxy,
/// then pick the arm with the highest mean.
///
/// Returns the `variant_id` of the selected arm, or `None` if `arms` is empty.
#[must_use]
pub fn thompson_sample(arms: &[PromptBanditArm]) -> Option<&str> {
    if arms.is_empty() {
        return None;
    }
    // Greedy on posterior mean = alpha / (alpha + beta).
    // Full Thompson sampling requires a Beta sampler; greedy-on-mean is a pragmatic
    // approximation sufficient for the bandit warm-start use case.
    arms.iter()
        .map(|arm| {
            let n = arm.alpha + arm.beta;
            let mean = arm.alpha / n;
            (mean, arm.variant_id.as_str())
        })
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, id)| id)
}

/// Check whether a variant has accumulated enough evidence to be promoted over the seed.
///
/// Returns `true` when:
/// - `n_tasks_total >= cfg.graduation_tasks`
/// - The variant's posterior mean exceeds the seed's mean by at least `cfg.promotion_margin`
#[must_use]
pub fn check_graduation(
    variant_id: &str,
    arms: &[PromptBanditArm],
    n_tasks_total: u32,
    cfg: &OproConfig,
) -> bool {
    if n_tasks_total < cfg.graduation_tasks {
        return false;
    }
    let Some(candidate) = arms.iter().find(|a| a.variant_id == variant_id) else {
        return false;
    };
    let Some(seed) = arms.iter().find(|a| a.variant_id == "seed") else {
        return false;
    };
    let candidate_mean = candidate.alpha / (candidate.alpha + candidate.beta);
    let seed_mean = seed.alpha / (seed.alpha + seed.beta);
    candidate_mean > seed_mean + cfg.promotion_margin
}

// ── Async OPRO trigger ────────────────────────────────────────────────────────

/// Run the OPRO trigger for an adapter after a task completes.
///
/// This is intended to be spawned as a background tokio task — errors are logged
/// but not propagated to the caller.
pub async fn run_opro_trigger<S: OproStore>(
    adapter_name: String,
    prompt_key: String,
    j_eff: f64,
    nats: &S,
    adapter: &dyn IComputeAdapter,
    cfg: &H2AIConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // 1. Load or init OPRO state.
    let mut state = nats
        .get_adapter_opro_state(&adapter_name)
        .await?
        .unwrap_or_else(|| AdapterOproState {
            adapter_name: adapter_name.clone(),
            j_eff_ema: j_eff,
            n_tasks_total: 0,
            n_tasks_since_last_opro: 0,
            last_opro_started_at: None,
            suppress_until_n_tasks: 0,
            bandit_arms: std::collections::HashMap::new(),
        });

    // 2. Update EMA.
    state.j_eff_ema = compute_ema(state.j_eff_ema, j_eff, cfg.opro.ema_window);
    state.n_tasks_total += 1;
    state.n_tasks_since_last_opro += 1;

    // 3. Check if OPRO should trigger.
    if should_trigger_opro(
        state.j_eff_ema,
        state.n_tasks_total,
        state.suppress_until_n_tasks,
        &cfg.opro,
    ) {
        // 4. Fetch current prompt text — prefer the active NATS variant; fall back to config.
        let current_prompt = {
            let active_id = nats
                .get_active_variant_ptr(&adapter_name, &prompt_key)
                .await?;

            let from_nats = if let Some(id) = active_id {
                nats.get_prompt_variant(&adapter_name, &prompt_key, &id)
                    .await?
                    .map(|v| v.text)
            } else {
                None
            };

            from_nats.unwrap_or_default()
        };

        // 5. Build OPRO meta-prompt and call LLM.
        let opro_system = "You are a prompt optimization assistant. \
                           Output only the improved prompt text, nothing else."
            .to_string();

        let opro_task = format!(
            "You are an expert prompt engineer. The current prompt for task '{}' is achieving \
             j_eff={:.2}. Please rewrite it to improve performance while keeping all template \
             variables like {{variable_name}} intact.\n\nCurrent prompt:\n{}\n\nImproved prompt:",
            prompt_key, state.j_eff_ema, current_prompt
        );

        let tau = TauValue::new(0.7).unwrap_or_else(|_| TauValue::new(0.5).unwrap());

        let request = ComputeRequest {
            system_context: opro_system,
            task: opro_task,
            tau,
            max_tokens: 2000,
        };

        let response = adapter.execute(request).await?;
        let candidate_text = response.output.trim().to_string();

        // 6. Validate — all template variables from the original must be present.
        if let Err(missing) = validate_opro_response(&current_prompt, &candidate_text) {
            tracing::warn!(
                adapter = %adapter_name,
                prompt_key = %prompt_key,
                ?missing,
                "OPRO response missing template variables; discarding candidate"
            );
            state.suppress_until_n_tasks = state.n_tasks_total + cfg.opro.suppress_n_tasks;
            nats.put_adapter_opro_state(&state).await?;
            return Ok(());
        }

        // 7. Store new variant.
        let variant_id = uuid::Uuid::new_v4().to_string();
        let variant = PromptVariant {
            variant_id: variant_id.clone(),
            adapter_name: adapter_name.clone(),
            prompt_key: prompt_key.clone(),
            text: candidate_text,
            source: PromptVariantSource::Opro,
            created_at: chrono::Utc::now(),
            score: None,
        };
        nats.put_prompt_variant(&variant).await?;

        tracing::info!(
            adapter = %adapter_name,
            prompt_key = %prompt_key,
            variant_id = %variant_id,
            j_eff_ema = state.j_eff_ema,
            "OPRO: stored new prompt variant"
        );

        // 8. Add bandit arm for the new variant (weak uniform prior).
        let arms = state.bandit_arms.entry(prompt_key.clone()).or_default();
        arms.push(PromptBanditArm {
            variant_id: variant_id.clone(),
            alpha: 1.0,
            beta: 1.0,
        });

        // 8b. Check graduation — promote the Thompson-sampled winner if criteria met.
        let arms_snapshot: Vec<PromptBanditArm> = state
            .bandit_arms
            .get(&prompt_key)
            .cloned()
            .unwrap_or_default();
        if let Some(winner_id) = thompson_sample(&arms_snapshot) {
            if check_graduation(winner_id, &arms_snapshot, state.n_tasks_total, &cfg.opro) {
                nats.set_active_variant_ptr(&adapter_name, &prompt_key, winner_id)
                    .await?;
                tracing::info!(
                    adapter = %adapter_name,
                    prompt_key = %prompt_key,
                    variant_id = %winner_id,
                    "OPRO: promoted variant via Thompson bandit graduation"
                );
            }
        }

        state.last_opro_started_at = Some(chrono::Utc::now());
        state.n_tasks_since_last_opro = 0;
        state.suppress_until_n_tasks = state.n_tasks_total + cfg.opro.suppress_n_tasks;
    }

    // 9. Persist updated state.
    nats.put_adapter_opro_state(&state).await?;
    Ok(())
}
