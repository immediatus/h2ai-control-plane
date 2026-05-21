use h2ai_config::H2AIConfig;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
#[allow(unused_imports)]
use h2ai_types::events::{LeaderElectedEvent, SocraticDiagnosisEvent};
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::TauValue;
use serde::{Deserialize, Serialize};

// ── Core types ────────────────────────────────────────────────────────────────

pub trait EpistemicLeader {
    type Belief;
    type Question;
    fn update_belief(&mut self, violations: &[String]) -> Self::Belief;
    fn formulate_question(&self, belief: &Self::Belief) -> Self::Question;
    fn should_rotate(&self) -> bool;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefRecord {
    pub question_hash: u64,
    pub question_text: String,
    pub outcome_scores: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderState {
    pub term: u32,
    pub leader_explorer_id: ExplorerId,
    pub prior_proposal: String,
    pub socratic_question: String,
    pub confidence_history: Vec<f64>,
    pub stagnation_count: u32,
    pub belief_buffer: Vec<BeliefRecord>,
    pub credibility_score: f64,
    pub follower_aspects: Vec<String>,
}

impl LeaderState {
    #[must_use]
    pub fn to_snapshot(&self, violated_constraints: Vec<String>) -> LeaderContextSnapshot {
        LeaderContextSnapshot {
            term: self.term,
            leader_explorer_id: self.leader_explorer_id.clone(),
            socratic_question: self.socratic_question.clone(),
            prior_proposal: self.prior_proposal.clone(),
            credibility_score: self.credibility_score,
            follower_aspects: self.follower_aspects.clone(),
            violated_constraints,
            belief_buffer_questions: self
                .belief_buffer
                .iter()
                .map(|r| r.question_text.clone())
                .collect(),
        }
    }
}

/// Slim read-only view passed through `PipelineParams` → `generation::Input`.
#[derive(Debug, Clone)]
pub struct LeaderContextSnapshot {
    pub term: u32,
    pub leader_explorer_id: ExplorerId,
    pub socratic_question: String,
    pub prior_proposal: String,
    pub credibility_score: f64,
    pub follower_aspects: Vec<String>,
    pub violated_constraints: Vec<String>,
    pub belief_buffer_questions: Vec<String>,
}

/// Data collected before the async diagnosis call, passed to `apply_leader_result`.
#[allow(dead_code)]
pub struct LeaderElectionPlan {
    pub task_id: TaskId,
    pub term: u32,
    pub leader_explorer_id: ExplorerId,
    pub runner_up_explorer_id: Option<ExplorerId>,
    pub prior_proposal: String,
    pub violated_constraint_ids: Vec<String>,
    pub q_confidence: f64,
    pub should_rotate: bool,
    pub follower_aspects: Vec<String>,
    pub existing_belief_buffer: Vec<BeliefRecord>,
    pub existing_credibility: f64,
}

// ── FNV-1a hash ───────────────────────────────────────────────────────────────

#[must_use]
pub fn fnv1a(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in s.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

// ── Helper functions ──────────────────────────────────────────────────────────

#[must_use]
pub fn should_rotate(state: &LeaderState, threshold: f64, waves: u32) -> bool {
    state.stagnation_count >= waves && state.confidence_history.len() >= 2 && {
        let n = state.confidence_history.len();
        let delta = state.confidence_history[n - 1] - state.confidence_history[n - 2];
        delta < threshold
    }
}

#[must_use]
pub fn update_credibility(current: f64, improved: bool, decay_rate: f64) -> f64 {
    if improved {
        (current + decay_rate).min(1.0)
    } else {
        (current - decay_rate).max(0.0)
    }
}

/// Heuristic EIG score: favours questions that mention more distinct constraint IDs
/// and haven't appeared in the belief buffer.
///
/// Returns 0.0 when the question hash exactly matches an existing buffer entry.
#[must_use]
pub fn eig_score(question: &str, constraint_ids: &[String], buffer: &[BeliefRecord]) -> f64 {
    let hash = fnv1a(question);
    if buffer.iter().any(|r| r.question_hash == hash) {
        return 0.0;
    }
    let mentioned = constraint_ids
        .iter()
        .filter(|id| question.contains(id.as_str()))
        .count() as f64;
    let diversity_bonus = 1.0
        - if buffer.is_empty() {
            0.0
        } else {
            let similar = buffer
                .iter()
                .filter(|r| {
                    let common = question
                        .split_whitespace()
                        .filter(|w| r.question_text.contains(*w))
                        .count();
                    common > 3
                })
                .count() as f64;
            (similar / buffer.len() as f64).min(1.0)
        };
    0.5f64.mul_add(diversity_bonus, mentioned)
}

/// Distribute violated constraint IDs to follower slots round-robin.
#[must_use]
pub fn assign_follower_aspects(constraint_ids: &[String], n_followers: usize) -> Vec<String> {
    if constraint_ids.is_empty() {
        return vec!["constraint resolution".to_string(); n_followers];
    }
    (0..n_followers)
        .map(|i| constraint_ids[i % constraint_ids.len()].clone())
        .collect()
}

/// Find the explorer with the highest verification score and the second-highest.
/// Returns `None` when `scores` is empty.
#[must_use]
pub fn select_best_and_runner_up(
    scores: &[(ExplorerId, f64)],
) -> Option<(ExplorerId, Option<ExplorerId>)> {
    if scores.is_empty() {
        return None;
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let winner = sorted[0].0.clone();
    let runner_up = sorted.get(1).map(|(id, _)| id.clone());
    Some((winner, runner_up))
}

// ── Async diagnosis ───────────────────────────────────────────────────────────

/// Call the adapter to generate EIG-ranked Socratic candidates.
///
/// Returns `(selected_question, eig_rank_1_based, n_dedup_candidates_tried)`.
pub async fn generate_socratic_question(
    adapter: &dyn IComputeAdapter,
    prior_proposal: &str,
    violated_constraints: &[String],
    belief_buffer: &[BeliefRecord],
    cfg: &H2AIConfig,
) -> (String, u32, u32) {
    let violation_list = violated_constraints.join(", ");
    let prior_questions: Vec<&str> = belief_buffer
        .iter()
        .map(|r| r.question_text.as_str())
        .collect();
    let prior_questions_block = if prior_questions.is_empty() {
        String::new()
    } else {
        format!(
            "\nQuestions you have already tried this session (do NOT repeat):\n{}",
            prior_questions.join("\n")
        )
    };

    let system_prompt = format!(
        "You are the epistemic leader for the next retry wave.\n\
         Your prior proposal follows.\n\
         Violated constraints: {violation_list}.\n\
         Formulate ONE Socratic question that challenges the core assumption \
         in your proposal — a question that, if answered differently, might resolve \
         the violations. Focus on the most uncertain causal node.\
         {prior_questions_block}\n\
         Output ONLY the question — no preamble, no explanation."
    );

    let n_candidates = cfg.leader_eig_candidates.max(1) as usize;
    let tau =
        TauValue::new(cfg.leader_diagnosis_tau).unwrap_or_else(|_| TauValue::new(0.3).unwrap());
    let proposal_snippet: String = prior_proposal.chars().take(4000).collect();

    let mut candidates: Vec<(String, f64)> = Vec::new();
    let mut dedup_tried: u32 = 0;

    for _ in 0..n_candidates {
        let req = ComputeRequest {
            system_context: system_prompt.clone(),
            task: proposal_snippet.clone(),
            tau,
            max_tokens: cfg.leader_diagnosis_max_tokens,
        };
        match adapter.execute(req).await {
            Ok(resp) => {
                let q = resp.output.trim().to_string();
                let hash = fnv1a(&q);
                let is_dup = belief_buffer.iter().any(|r| r.question_hash == hash);
                if is_dup {
                    dedup_tried += 1;
                    continue;
                }
                let score = eig_score(&q, violated_constraints, belief_buffer);
                candidates.push((q, score));
            }
            Err(_) => continue,
        }
    }

    if candidates.is_empty() {
        let fallback = violated_constraints.first().map_or_else(
            || "What core assumption might be preventing constraint satisfaction?".to_string(),
            |id| format!("What if the approach to {id} is fundamentally wrong?"),
        );
        return (fallback, 1, dedup_tried);
    }

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let eig_rank = 1u32;
    (candidates.remove(0).0, eig_rank, dedup_tried)
}

// ── Context prefix builders ───────────────────────────────────────────────────

/// Build the per-slot context prefix for the leader slot.
#[must_use]
pub fn build_leader_prefix(snapshot: &LeaderContextSnapshot, explorer_id: &ExplorerId) -> String {
    let term = snapshot.term;
    let question = &snapshot.socratic_question;
    let credibility = snapshot.credibility_score;

    if *explorer_id == snapshot.leader_explorer_id {
        let prior = snapshot
            .prior_proposal
            .chars()
            .take(3000)
            .collect::<String>();
        let past_q_block = if snapshot.belief_buffer_questions.is_empty() {
            String::new()
        } else {
            format!(
                "\nQuestions you have already tried this session (avoid these angles):\n{}",
                snapshot.belief_buffer_questions.join("\n")
            )
        };
        format!(
            "\n--- LEADER CONTEXT (term {term}) ---\n\
             You are the current leader (credibility: {credibility:.2}).\n\
             Your Socratic question for this wave:\n\
             \"{question}\"\n\
             \n\
             Your goal: answer your own question better than before.\n\
             Do not repeat your prior approach verbatim.\
             {past_q_block}\n\
             \n\
             Violated constraints: {violated}.\n\
             \n\
             Your prior proposal:\n\
             {prior}\n\
             --- END LEADER CONTEXT ---\n",
            violated = snapshot.violated_constraints.join(", "),
        )
    } else {
        format!(
            "\n--- FOLLOWER CONTEXT (term {term}) ---\n\
             The leader's diagnostic question for this wave:\n\
             \"{question}\"\n\
             \n\
             Treat this as an open question. Form your own independent answer \
             — do not defer to the leader's prior approach.\n\
             Explore a genuinely different resolution.\n\
             --- END FOLLOWER CONTEXT ---\n"
        )
    }
}

/// Build the per-slot prefix with aspect specialisation, given follower slot index.
#[must_use]
pub fn build_follower_prefix_with_aspect(
    snapshot: &LeaderContextSnapshot,
    slot_index: usize,
    warn_threshold: f64,
) -> String {
    let term = snapshot.term;
    let question = &snapshot.socratic_question;
    let credibility = snapshot.credibility_score;

    let aspect = snapshot
        .follower_aspects
        .get(slot_index)
        .cloned()
        .unwrap_or_else(|| "constraint resolution".to_string());

    let low_conf_prefix = if credibility < warn_threshold {
        format!(
            "[Note: leader signal is low-confidence (score={credibility:.2}). \
             Treat as a weak hint, not a directive.]\n"
        )
    } else {
        String::new()
    };

    format!(
        "\n{low_conf_prefix}\
         --- FOLLOWER CONTEXT (term {term}) ---\n\
         The leader's diagnostic question for this wave:\n\
         \"{question}\"\n\
         \n\
         Your assigned aspect to probe: {aspect}\n\
         \n\
         Treat this as an open question. Form your own independent answer \
         — do not defer to the leader's prior approach.\n\
         Focus specifically on your assigned aspect; the other followers \
         will cover the remaining dimensions.\n\
         --- END FOLLOWER CONTEXT ---\n"
    )
}
