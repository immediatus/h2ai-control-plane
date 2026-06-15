//! GAP-F6: Plan-Awareness Probe — batched LLM judge between Stage 1 and Stage 2.

use std::collections::HashMap;

use crate::llm_parse::{extract_first_json_array, strip_json_fences};
use h2ai_constraints::ambiguity::{
    scan_constraint, score_evidence, AmbiguityDetectionConfig, AmbiguityScorecard,
};
use h2ai_constraints::types::{ConstraintDoc, ConstraintSeverity};

// ── Verdict types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ProbeVerdict {
    Acknowledged,
    NotAddressed,
    Contradicted,
}

/// One item submitted to the judge (one constraint).
#[derive(Debug, Clone)]
pub struct ProbeItem {
    pub constraint_id: String,
    /// Formatted: "[{id}] ({Hard|Soft})\n{description}\nPASS CRITERIA: {criteria}"
    pub text: String,
    pub is_hard: bool,
    /// True when `is_ambiguity_gated` fires — verdict is recorded but can never block.
    pub gated: bool,
}

/// One judged constraint as returned by the judge (raw wire form; rationale precedes verdict per R2).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ConstraintVerdict {
    pub idx: usize,
    pub rationale: String,
    pub verdict: ProbeVerdict,
}

/// One judged constraint after matching back to its ProbeItem.
#[derive(Debug, Clone)]
pub struct ProbeOutcome {
    pub constraint_id: String,
    pub verdict: ProbeVerdict,
    pub rationale: String,
    pub is_hard: bool,
    pub gated: bool,
}

/// Result of a full probe run.
#[derive(Debug, Clone)]
pub struct ProbeResult {
    pub outcomes: Vec<ProbeOutcome>,
    pub n_items: usize,
    /// Items the judge did not return a verdict for (truncation / malformed entries).
    /// n_unjudged > 0 ⇒ degraded.
    pub n_unjudged: usize,
    /// True when judge call failed entirely, or n_unjudged > 0.
    pub degraded: bool,
}

impl ProbeResult {
    /// Hard, non-gated CONTRADICTED outcomes — the only blocking class.
    pub fn blocking(&self) -> Vec<&ProbeOutcome> {
        self.outcomes
            .iter()
            .filter(|o| o.verdict == ProbeVerdict::Contradicted && o.is_hard && !o.gated)
            .collect()
    }

    /// Hint text for the thinking-loop re-iteration. Returns `None` when:
    /// - result is degraded (parse/truncation failure)
    /// - no Hard, non-gated CONTRADICTED outcomes exist
    /// - mode is Shadow (caller decides; this method only checks the result)
    pub fn re_iteration_prompt(&self) -> Option<String> {
        if self.degraded {
            return None;
        }
        let blocking = self.blocking();
        if blocking.is_empty() {
            return None;
        }
        let hints = blocking
            .iter()
            .map(|o| {
                format!(
                    "• [{}] Your plan contradicts this requirement: {}",
                    o.constraint_id, o.rationale
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        Some(format!(
            "## Constraint contradiction check\n\
             Your plan affirmatively contradicts the following hard requirements. \
             Revise the approach to satisfy each:\n\n{hints}"
        ))
    }
}

// ── Static ambiguity gate ──────────────────────────────────────────────────────

/// Returns true when the constraint's static evidence alone reaches `cfg.score_threshold`.
/// Uses the same `scan_constraint` + `score_evidence` pair as GAP-F8's `seed_scorecards`.
///
/// AmbiguityScorecard has no Default — construct with `AmbiguityScorecard::new`.
/// score_evidence returns a new scorecard; assign the result back through the mutable reference.
pub fn is_ambiguity_gated(doc: &ConstraintDoc, cfg: &AmbiguityDetectionConfig) -> bool {
    if !cfg.enabled {
        return false;
    }
    let mut by_check: HashMap<usize, AmbiguityScorecard> = HashMap::new();
    for (check_idx, evidence) in scan_constraint(doc) {
        let card = by_check
            .entry(check_idx)
            .or_insert_with(|| AmbiguityScorecard::new(doc.id.clone(), check_idx));
        let updated = score_evidence(card, evidence, cfg);
        *card = updated;
    }
    by_check
        .values()
        .any(|card| card.score >= cfg.score_threshold)
}

// ── Probe item builder ─────────────────────────────────────────────────────────

/// Build probe items from the constraint corpus.
/// - Advisory constraints are excluded entirely.
/// - `pass_criteria` is used as the criteria text; falls back to `description` when absent.
/// - Ambiguity-gated constraints are included but marked `gated: true`.
pub fn build_probe_items(
    constraints: &[ConstraintDoc],
    ambiguity_cfg: &AmbiguityDetectionConfig,
) -> Vec<ProbeItem> {
    constraints
        .iter()
        .filter(|doc| !matches!(doc.severity, ConstraintSeverity::Advisory))
        .map(|doc| {
            let severity_label = if matches!(doc.severity, ConstraintSeverity::Hard { .. }) {
                "Hard"
            } else {
                "Soft"
            };
            let criteria = doc.pass_criteria.as_deref().unwrap_or(&doc.description);
            let text = format!(
                "[{}] ({})\n{}\nPASS CRITERIA: {}",
                doc.id, severity_label, doc.description, criteria,
            );
            ProbeItem {
                constraint_id: doc.id.clone(),
                text,
                is_hard: matches!(doc.severity, ConstraintSeverity::Hard { .. }),
                gated: is_ambiguity_gated(doc, ambiguity_cfg),
            }
        })
        .collect()
}

// ── Judge trait ────────────────────────────────────────────────────────────────

#[async_trait::async_trait]
pub trait AwarenessJudge: Send + Sync {
    /// One batched call. Returns `None` on call or whole-parse failure.
    async fn judge(
        &self,
        understanding: &str,
        items: &[ProbeItem],
    ) -> Option<Vec<ConstraintVerdict>>;
}

// ── LLM judge (production) ─────────────────────────────────────────────────────

pub struct LlmAwarenessJudge {
    adapter: std::sync::Arc<dyn h2ai_types::adapter::IComputeAdapter>,
    max_tokens: u64,
}

impl LlmAwarenessJudge {
    pub fn new(
        adapter: std::sync::Arc<dyn h2ai_types::adapter::IComputeAdapter>,
        max_tokens: u64,
    ) -> Self {
        Self {
            adapter,
            max_tokens,
        }
    }
}

#[async_trait::async_trait]
impl AwarenessJudge for LlmAwarenessJudge {
    async fn judge(
        &self,
        understanding: &str,
        items: &[ProbeItem],
    ) -> Option<Vec<ConstraintVerdict>> {
        use h2ai_types::adapter::ComputeRequest;
        use h2ai_types::sizing::TauValue;

        let constraints_block = items
            .iter()
            .enumerate()
            .map(|(i, item)| format!("{}. {}", i, item.text))
            .collect::<Vec<_>>()
            .join("\n\n");

        let request = ComputeRequest {
            system_context: "You are a strict design reviewer. You will receive a plan \
                and a numbered list of constraints (each with pass criteria). For EACH \
                constraint, first write a one-sentence rationale citing plan content, \
                then give a verdict. Respond with ONLY a JSON array:\n\
                [{\"idx\": 0, \"rationale\": \"...\", \"verdict\": \"ACKNOWLEDGED\"}, ...]\n\
                Verdicts: \"ACKNOWLEDGED\" (plan demonstrates awareness of the invariant), \
                \"NOT_ADDRESSED\" (invariant not mentioned), \
                \"CONTRADICTED\" (plan affirmatively proposes something that violates the \
                pass criteria — cite the contradicting plan content)."
                .to_string(),
            task: format!("PLAN:\n{understanding}\n\nCONSTRAINTS:\n{constraints_block}"),
            tau: TauValue::new(0.1).unwrap(),
            max_tokens: self.max_tokens,
        };

        let response = self.adapter.execute(request).await.ok()?;
        parse_probe_verdicts(&response.output)
    }
}

// ── Parse helper ───────────────────────────────────────────────────────────────

/// Parse the judge's raw output text into `ConstraintVerdict` entries.
/// Uses `llm_parse` helpers for fence-stripping and preamble handling.
/// Malformed individual items are silently dropped (they surface as `n_unjudged`).
/// Returns `None` only on whole-parse failure (no valid JSON array found at all).
/// An empty array `[]` returns `Some(vec![])` — distinct from failure.
pub fn parse_probe_verdicts(text: &str) -> Option<Vec<ConstraintVerdict>> {
    let stripped = strip_json_fences(text);
    let array_str = extract_first_json_array(stripped)?;
    let arr: Vec<serde_json::Value> = serde_json::from_str(array_str).ok()?;
    let verdicts: Vec<ConstraintVerdict> = arr
        .into_iter()
        .filter_map(|v| serde_json::from_value(v).ok())
        .collect();
    Some(verdicts)
}

// ── Core probe runner ──────────────────────────────────────────────────────────

/// Run the probe. Deterministic given judge responses.
/// All failure modes (call failure, partial array, out-of-range idx) set `degraded = true`
/// and cause `re_iteration_prompt()` to return `None` — degraded probes never block.
pub async fn run_awareness_probe(
    understanding: &str,
    items: &[ProbeItem],
    judge: &dyn AwarenessJudge,
) -> ProbeResult {
    if items.is_empty() {
        return ProbeResult {
            outcomes: vec![],
            n_items: 0,
            n_unjudged: 0,
            degraded: false,
        };
    }
    let Some(verdicts) = judge.judge(understanding, items).await else {
        // Whole call or parse failure
        return ProbeResult {
            outcomes: vec![],
            n_items: items.len(),
            n_unjudged: items.len(),
            degraded: true,
        };
    };

    let mut outcomes: Vec<ProbeOutcome> = Vec::new();
    let mut judged = vec![false; items.len()];

    for v in &verdicts {
        let Some(item) = items.get(v.idx) else {
            continue;
        }; // out-of-range idx dropped
        if judged[v.idx] {
            continue;
        } // drop duplicate idx silently
        judged[v.idx] = true;
        outcomes.push(ProbeOutcome {
            constraint_id: item.constraint_id.clone(),
            verdict: v.verdict,
            rationale: v.rationale.clone(),
            is_hard: item.is_hard,
            gated: item.gated,
        });
    }

    let n_unjudged = judged.iter().filter(|j| !**j).count();
    ProbeResult {
        outcomes,
        n_items: items.len(),
        n_unjudged,
        degraded: n_unjudged > 0,
    }
}
