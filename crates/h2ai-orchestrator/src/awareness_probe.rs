//! GAP-F6: Plan-Awareness Probe — batched LLM judge between Stage 1 and Stage 2.

use std::collections::HashMap;

use crate::llm_parse::{extract_first_json_array, strip_json_fences};
use h2ai_constraints::ambiguity::{
    score_evidence, scan_constraint, AmbiguityDetectionConfig, AmbiguityScorecard,
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
            .filter(|o| {
                o.verdict == ProbeVerdict::Contradicted && o.is_hard && !o.gated
            })
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
    by_check.values().any(|card| card.score >= cfg.score_threshold)
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
            let severity_label =
                if matches!(doc.severity, ConstraintSeverity::Hard { .. }) {
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

// ── Mock judge (unit tests only) ──────────────────────────────────────────────

#[cfg(test)]
pub struct MockAwarenessJudge {
    /// `Some(vec)` → returns those verdicts; `None` → simulates call failure.
    pub verdicts: Option<Vec<ConstraintVerdict>>,
}

#[cfg(test)]
#[async_trait::async_trait]
impl AwarenessJudge for MockAwarenessJudge {
    async fn judge(
        &self,
        _understanding: &str,
        _items: &[ProbeItem],
    ) -> Option<Vec<ConstraintVerdict>> {
        self.verdicts.clone()
    }
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
        Self { adapter, max_tokens }
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
        let Some(item) = items.get(v.idx) else { continue }; // out-of-range idx dropped
        if judged[v.idx] { continue } // drop duplicate idx silently
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

#[cfg(test)]
mod tests {
    use super::*;
    use h2ai_constraints::ambiguity::AmbiguityDetectionConfig;
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};

    fn hard_doc(id: &str, pass: &str) -> ConstraintDoc {
        ConstraintDoc {
            id: id.to_string(),
            source_file: "test.yaml".to_string(),
            description: format!("{id} description"),
            severity: ConstraintSeverity::Hard { threshold: 0.7 },
            predicate: ConstraintPredicate::LlmJudge {
                rubric: "always pass".to_string(),
            },
            remediation_hint: None,
            domains: vec!["billing".to_string()],
            mandatory_for_tags: vec![],
            related_to: vec![],
            binary_checks: vec![],
            version: 1,
            repair_provenance: None,
            pass_criteria: Some(pass.to_string()),
        }
    }

    fn soft_doc(id: &str, pass: &str) -> ConstraintDoc {
        let mut d = hard_doc(id, pass);
        d.severity = ConstraintSeverity::Soft { weight: 0.5 };
        d
    }

    fn advisory_doc(id: &str) -> ConstraintDoc {
        let mut d = hard_doc(id, "any");
        d.severity = ConstraintSeverity::Advisory;
        d
    }

    fn no_pass_criteria_doc(id: &str) -> ConstraintDoc {
        let mut d = hard_doc(id, "ignored");
        d.pass_criteria = None;
        d
    }

    /// CONSTRAINT-005-shaped doc: MultiStorageConflict + FmTermNegation + RemediationConflict
    /// push check 0's accumulated score above 0.6. Used to test ambiguity gating (finding #4).
    fn ambiguous_hard_doc(id: &str) -> ConstraintDoc {
        let rubric = "Does the proposal use a dual-ledger model: CockroachDB for operational \
                      state, ClickHouse for immutable audit?\n\
                      FM: Avoid CockroachDB on the synchronous charge path — latency budget."
            .to_string();
        ConstraintDoc {
            id: id.to_string(),
            source_file: "test.yaml".to_string(),
            description: format!("{id} description"),
            severity: ConstraintSeverity::Hard { threshold: 0.7 },
            predicate: ConstraintPredicate::LlmJudge { rubric },
            remediation_hint: Some(
                "Use Redis for the hot ledger and append-only ClickHouse for audit.".to_string(),
            ),
            domains: vec!["billing".to_string()],
            mandatory_for_tags: vec![],
            related_to: vec![],
            binary_checks: vec![
                "Does the proposal use a dual-ledger model: CockroachDB for operational state, \
                 ClickHouse for immutable audit?"
                    .to_string(),
            ],
            version: 1,
            repair_provenance: None,
            pass_criteria: Some(format!("{id} pass criteria")),
        }
    }

    fn enabled_ambiguity_cfg() -> AmbiguityDetectionConfig {
        AmbiguityDetectionConfig {
            enabled: true,
            ..AmbiguityDetectionConfig::default()
        }
    }

    #[test]
    fn build_items_excludes_advisory() {
        let docs = vec![
            hard_doc("C-1", "use atomic lua"),
            advisory_doc("C-2"),
            soft_doc("C-3", "log all events"),
        ];
        let cfg = AmbiguityDetectionConfig::default();
        let items = build_probe_items(&docs, &cfg);
        assert_eq!(items.len(), 2);
        assert!(items.iter().all(|i| i.constraint_id != "C-2"));
    }

    #[test]
    fn build_items_is_hard_flag() {
        let docs = vec![hard_doc("C-1", "p"), soft_doc("C-2", "q")];
        let cfg = AmbiguityDetectionConfig::default();
        let items = build_probe_items(&docs, &cfg);
        assert!(items.iter().find(|i| i.constraint_id == "C-1").unwrap().is_hard);
        assert!(!items.iter().find(|i| i.constraint_id == "C-2").unwrap().is_hard);
    }

    #[test]
    fn build_items_falls_back_to_description_when_no_pass_criteria() {
        let doc = no_pass_criteria_doc("C-1");
        let cfg = AmbiguityDetectionConfig::default();
        let items = build_probe_items(&[doc.clone()], &cfg);
        assert_eq!(items.len(), 1);
        // text must contain the description (used as fallback)
        assert!(items[0].text.contains(&doc.description));
    }

    #[test]
    fn build_items_text_contains_pass_criteria() {
        let doc = hard_doc("C-1", "must use Lua EVAL atomically");
        let cfg = AmbiguityDetectionConfig::default();
        let items = build_probe_items(&[doc], &cfg);
        assert!(items[0].text.contains("must use Lua EVAL atomically"));
    }

    #[test]
    fn re_iteration_prompt_none_when_degraded() {
        let result = ProbeResult {
            outcomes: vec![ProbeOutcome {
                constraint_id: "C-1".into(),
                verdict: ProbeVerdict::Contradicted,
                rationale: "plan uses non-atomic write".into(),
                is_hard: true,
                gated: false,
            }],
            n_items: 1,
            n_unjudged: 1,
            degraded: true,
        };
        assert!(result.re_iteration_prompt().is_none());
    }

    #[test]
    fn re_iteration_prompt_none_when_no_blockers() {
        let result = ProbeResult {
            outcomes: vec![ProbeOutcome {
                constraint_id: "C-1".into(),
                verdict: ProbeVerdict::Acknowledged,
                rationale: "plan mentions Lua".into(),
                is_hard: true,
                gated: false,
            }],
            n_items: 1,
            n_unjudged: 0,
            degraded: false,
        };
        assert!(result.re_iteration_prompt().is_none());
    }

    #[test]
    fn re_iteration_prompt_cites_rationale() {
        let result = ProbeResult {
            outcomes: vec![ProbeOutcome {
                constraint_id: "C-1".into(),
                verdict: ProbeVerdict::Contradicted,
                rationale: "plan uses non-atomic write".into(),
                is_hard: true,
                gated: false,
            }],
            n_items: 1,
            n_unjudged: 0,
            degraded: false,
        };
        let prompt = result.re_iteration_prompt().expect("must produce hint");
        assert!(prompt.contains("C-1"));
        assert!(prompt.contains("non-atomic write"));
    }

    #[test]
    fn soft_contradicted_does_not_block() {
        let result = ProbeResult {
            outcomes: vec![ProbeOutcome {
                constraint_id: "C-1".into(),
                verdict: ProbeVerdict::Contradicted,
                rationale: "missing log".into(),
                is_hard: false,
                gated: false,
            }],
            n_items: 1,
            n_unjudged: 0,
            degraded: false,
        };
        assert!(result.re_iteration_prompt().is_none());
    }

    #[test]
    fn gated_contradicted_does_not_block() {
        let result = ProbeResult {
            outcomes: vec![ProbeOutcome {
                constraint_id: "C-1".into(),
                verdict: ProbeVerdict::Contradicted,
                rationale: "redis vs cockroachdb contradiction".into(),
                is_hard: true,
                gated: true,
            }],
            n_items: 1,
            n_unjudged: 0,
            degraded: false,
        };
        assert!(result.re_iteration_prompt().is_none());
    }

    #[test]
    fn is_ambiguity_gated_disabled_config_never_gates() {
        // When ambiguity detection is disabled, no constraint should be gated
        // regardless of its binary_checks content.
        let mut cfg = AmbiguityDetectionConfig::default();
        cfg.enabled = false;
        let doc = hard_doc("C-1", "use atomic lua");
        // Even a doc with many binary_checks should not be gated when disabled.
        assert!(!is_ambiguity_gated(&doc, &cfg));
    }

    #[test]
    fn build_items_marks_statically_ambiguous_as_gated() {
        // Finding #4: CONSTRAINT-005-shaped constraint (MultiStorageConflict +
        // FmTermNegation + RemediationConflict on check 0) must surface as gated: true
        // so it can never trigger re-iteration even in active mode.
        let docs = vec![hard_doc("C-1", "use atomic lua"), ambiguous_hard_doc("C-AMB")];
        let cfg = enabled_ambiguity_cfg();
        let items = build_probe_items(&docs, &cfg);
        assert_eq!(items.len(), 2, "advisory exclusion must not affect non-advisory docs");
        let plain = items.iter().find(|i| i.constraint_id == "C-1").unwrap();
        let gated = items.iter().find(|i| i.constraint_id == "C-AMB").unwrap();
        assert!(!plain.gated, "non-ambiguous hard constraint must not be gated");
        assert!(
            gated.gated,
            "CONSTRAINT-005-shaped constraint must be gated (finding #4 safety invariant)"
        );
    }

    #[test]
    fn parse_probe_verdicts_with_think_tag_preamble() {
        // R4 regression: DeepSeek-style models emit <think>…</think> before the JSON array.
        // extract_first_json_array must find the array even with arbitrary XML preamble.
        let raw = "<think>Let me evaluate each constraint carefully against the plan.</think>\n\
                   [{\"idx\":0,\"rationale\":\"plan explicitly uses Lua EVAL\",\"verdict\":\"ACKNOWLEDGED\"}]";
        let v = parse_probe_verdicts(raw).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].verdict, ProbeVerdict::Acknowledged);
        assert_eq!(v[0].idx, 0);
    }

    // ── Judge trait + run_awareness_probe tests ────────────────────────────────

    fn verdict(idx: usize, v: ProbeVerdict) -> ConstraintVerdict {
        ConstraintVerdict {
            idx,
            rationale: format!("rationale-for-{idx}"),
            verdict: v,
        }
    }

    fn make_items(n: usize) -> Vec<ProbeItem> {
        (0..n)
            .map(|i| ProbeItem {
                constraint_id: format!("C-{i}"),
                text: format!("item-{i}"),
                is_hard: true,
                gated: false,
            })
            .collect()
    }

    #[tokio::test]
    async fn probe_empty_items_returns_clean_result() {
        let judge = MockAwarenessJudge { verdicts: Some(vec![]) };
        let result = run_awareness_probe("understanding", &[], &judge).await;
        assert_eq!(result.n_items, 0);
        assert_eq!(result.n_unjudged, 0);
        assert!(!result.degraded);
        assert!(result.outcomes.is_empty());
    }

    #[tokio::test]
    async fn probe_judge_failure_marks_all_unjudged_and_degraded() {
        let judge = MockAwarenessJudge { verdicts: None };
        let items = make_items(3);
        let result = run_awareness_probe("understanding", &items, &judge).await;
        assert!(result.degraded);
        assert_eq!(result.n_unjudged, 3);
        assert!(result.re_iteration_prompt().is_none());
    }

    #[tokio::test]
    async fn probe_partial_verdicts_degrade() {
        // 4 items, judge returns only 2 → n_unjudged = 2, degraded = true
        let judge = MockAwarenessJudge {
            verdicts: Some(vec![
                verdict(0, ProbeVerdict::Contradicted),
                verdict(1, ProbeVerdict::Acknowledged),
            ]),
        };
        let items = make_items(4);
        let result = run_awareness_probe("understanding", &items, &judge).await;
        assert!(result.degraded);
        assert_eq!(result.n_unjudged, 2);
        assert!(result.re_iteration_prompt().is_none()); // degraded → never blocks
    }

    #[tokio::test]
    async fn probe_all_acknowledged_no_blockers() {
        let items = make_items(2);
        let judge = MockAwarenessJudge {
            verdicts: Some(vec![
                verdict(0, ProbeVerdict::Acknowledged),
                verdict(1, ProbeVerdict::Acknowledged),
            ]),
        };
        let result = run_awareness_probe("understanding", &items, &judge).await;
        assert!(!result.degraded);
        assert_eq!(result.n_unjudged, 0);
        assert!(result.re_iteration_prompt().is_none());
    }

    #[tokio::test]
    async fn probe_contradicted_hard_non_gated_produces_hint() {
        let items = vec![ProbeItem {
            constraint_id: "C-1".into(),
            text: "item".into(),
            is_hard: true,
            gated: false,
        }];
        let judge = MockAwarenessJudge {
            verdicts: Some(vec![verdict(0, ProbeVerdict::Contradicted)]),
        };
        let result = run_awareness_probe("plan text", &items, &judge).await;
        assert!(!result.degraded);
        assert!(result.re_iteration_prompt().is_some());
    }

    #[tokio::test]
    async fn probe_not_addressed_never_blocks() {
        let items = make_items(1);
        let judge = MockAwarenessJudge {
            verdicts: Some(vec![verdict(0, ProbeVerdict::NotAddressed)]),
        };
        let result = run_awareness_probe("understanding", &items, &judge).await;
        assert!(!result.degraded);
        assert!(result.re_iteration_prompt().is_none());
    }

    #[tokio::test]
    async fn probe_out_of_range_idx_counts_as_unjudged() {
        let items = make_items(2); // indices 0, 1 valid
        let judge = MockAwarenessJudge {
            verdicts: Some(vec![
                verdict(0, ProbeVerdict::Acknowledged),
                verdict(99, ProbeVerdict::Contradicted), // out of range
            ]),
        };
        let result = run_awareness_probe("understanding", &items, &judge).await;
        // idx 99 is dropped; item 1 was never judged → n_unjudged = 1 → degraded
        assert!(result.degraded);
        assert_eq!(result.n_unjudged, 1);
    }

    #[test]
    fn parse_probe_verdicts_with_fenced_json() {
        let raw = "```json\n[{\"idx\":0,\"rationale\":\"ok\",\"verdict\":\"ACKNOWLEDGED\"}]\n```";
        let v = parse_probe_verdicts(raw).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].verdict, ProbeVerdict::Acknowledged);
    }

    #[test]
    fn parse_probe_verdicts_with_preamble() {
        // Regression: local models emit preamble text before the array
        let raw = "Here are my verdicts:\n[{\"idx\":0,\"rationale\":\"ok\",\"verdict\":\"CONTRADICTED\"}]";
        let v = parse_probe_verdicts(raw).unwrap();
        assert_eq!(v[0].verdict, ProbeVerdict::Contradicted);
    }

    #[test]
    fn parse_probe_verdicts_garbage_returns_none() {
        assert!(parse_probe_verdicts("not json at all").is_none());
    }

    #[test]
    fn parse_probe_verdicts_malformed_item_dropped() {
        // One valid item, one with wrong verdict string → only valid one survives
        let raw = r#"[{"idx":0,"rationale":"ok","verdict":"ACKNOWLEDGED"},{"idx":1,"rationale":"x","verdict":"INVALID"}]"#;
        let v = parse_probe_verdicts(raw).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].idx, 0);
    }

    #[test]
    fn parse_probe_verdicts_empty_array_returns_empty_vec() {
        let v = parse_probe_verdicts("[]").unwrap();
        assert!(v.is_empty());
    }

    #[tokio::test]
    async fn probe_duplicate_idx_does_not_produce_duplicate_outcomes() {
        let items = make_items(1);
        let judge = MockAwarenessJudge {
            verdicts: Some(vec![
                verdict(0, ProbeVerdict::Contradicted),
                verdict(0, ProbeVerdict::Acknowledged), // duplicate — second must be dropped
            ]),
        };
        let result = run_awareness_probe("understanding", &items, &judge).await;
        assert_eq!(result.outcomes.len(), 1);
        assert!(!result.degraded);
        // First verdict wins (Contradicted)
        assert_eq!(result.outcomes[0].verdict, ProbeVerdict::Contradicted);
    }

    #[tokio::test]
    async fn probe_empty_verdicts_for_non_empty_items_degrades() {
        let judge = MockAwarenessJudge { verdicts: Some(vec![]) };
        let result = run_awareness_probe("understanding", &make_items(3), &judge).await;
        assert!(result.degraded);
        assert_eq!(result.n_unjudged, 3);
    }

    #[tokio::test]
    async fn llm_judge_calls_adapter_and_parses_response() {
        use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse};
        use h2ai_types::config::{AdapterKind, CloudProvider};

        #[derive(Debug)]
        struct FakeAdapter {
            response: String,
            kind: AdapterKind,
        }

        #[async_trait::async_trait]
        impl h2ai_types::adapter::IComputeAdapter for FakeAdapter {
            async fn execute(
                &self,
                _req: ComputeRequest,
            ) -> Result<ComputeResponse, AdapterError> {
                Ok(ComputeResponse {
                    output: self.response.clone(),
                    token_cost: 0,
                    adapter_kind: self.kind.clone(),
                    tokens_used: None,
                    reasoning_trace: None,
                })
            }
            fn kind(&self) -> &AdapterKind {
                &self.kind
            }
        }

        let response_text = r#"[{"idx":0,"rationale":"plan mentions Lua atomicity","verdict":"ACKNOWLEDGED"}]"#;
        let adapter = std::sync::Arc::new(FakeAdapter {
            response: response_text.to_string(),
            kind: AdapterKind::CloudGeneric {
                endpoint: "http://fake".into(),
                api_key_env: "FAKE_KEY".into(),
                model: None,
                provider: CloudProvider::default(),
            },
        });

        let judge = LlmAwarenessJudge::new(adapter, 512);
        let items = vec![ProbeItem {
            constraint_id: "C-1".into(),
            text: "C-1 pass criteria text".into(),
            is_hard: true,
            gated: false,
        }];
        let result = judge.judge("plan uses Lua EVAL", &items).await;
        let verdicts = result.expect("must return verdicts");
        assert_eq!(verdicts.len(), 1);
        assert_eq!(verdicts[0].verdict, ProbeVerdict::Acknowledged);
    }

    #[tokio::test]
    async fn llm_judge_adapter_error_returns_none() {
        use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse};
        use h2ai_types::config::{AdapterKind, CloudProvider};

        #[derive(Debug)]
        struct FailAdapter;

        #[async_trait::async_trait]
        impl h2ai_types::adapter::IComputeAdapter for FailAdapter {
            async fn execute(
                &self,
                _req: ComputeRequest,
            ) -> Result<ComputeResponse, AdapterError> {
                Err(AdapterError::Timeout)
            }
            fn kind(&self) -> &AdapterKind {
                static KIND: std::sync::OnceLock<AdapterKind> = std::sync::OnceLock::new();
                KIND.get_or_init(|| AdapterKind::CloudGeneric {
                    endpoint: "http://fail".into(),
                    api_key_env: "FAIL_KEY".into(),
                    model: None,
                    provider: CloudProvider::default(),
                })
            }
        }

        let judge = LlmAwarenessJudge::new(std::sync::Arc::new(FailAdapter), 512);
        let items = vec![ProbeItem {
            constraint_id: "C-1".into(),
            text: "text".into(),
            is_hard: true,
            gated: false,
        }];
        let result = judge.judge("plan", &items).await;
        assert!(result.is_none(), "adapter error must propagate as None");
    }

    #[tokio::test]
    async fn llm_judge_request_contains_plan_and_constraints() {
        use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse};
        use h2ai_types::config::{AdapterKind, CloudProvider};
        use std::sync::Mutex;

        struct CapturingAdapter {
            captured: Mutex<Option<ComputeRequest>>,
            kind: AdapterKind,
        }

        impl std::fmt::Debug for CapturingAdapter {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("CapturingAdapter").finish()
            }
        }

        #[async_trait::async_trait]
        impl h2ai_types::adapter::IComputeAdapter for CapturingAdapter {
            async fn execute(
                &self,
                req: ComputeRequest,
            ) -> Result<ComputeResponse, AdapterError> {
                *self.captured.lock().unwrap() = Some(req);
                Ok(ComputeResponse {
                    output: "[]".to_string(),
                    token_cost: 0,
                    adapter_kind: self.kind.clone(),
                    tokens_used: None,
                    reasoning_trace: None,
                })
            }
            fn kind(&self) -> &AdapterKind {
                &self.kind
            }
        }

        let capturing = std::sync::Arc::new(CapturingAdapter {
            captured: Mutex::new(None),
            kind: AdapterKind::CloudGeneric {
                endpoint: "http://capture".into(),
                api_key_env: "CAP_KEY".into(),
                model: None,
                provider: CloudProvider::default(),
            },
        });
        let adapter_arc: std::sync::Arc<dyn h2ai_types::adapter::IComputeAdapter> =
            std::sync::Arc::clone(&capturing) as _;
        let judge = LlmAwarenessJudge::new(adapter_arc, 2048);
        let items = vec![ProbeItem {
            constraint_id: "C-1".into(),
            text: "use atomic Lua".into(),
            is_hard: true,
            gated: false,
        }];
        let _ = judge.judge("the plan text", &items).await;

        let req = capturing.captured.lock().unwrap().take().expect("adapter must be called");
        assert!(req.task.contains("PLAN:\nthe plan text"), "task must start with PLAN:");
        assert!(req.task.contains("CONSTRAINTS:\n"), "task must contain CONSTRAINTS:");
        assert!(req.task.contains("use atomic Lua"), "task must contain constraint text");
        assert_eq!(req.max_tokens, 2048, "max_tokens must match constructor arg");
        // tau = 0.1 (deterministic classification)
        assert!((req.tau.value() - 0.1_f64).abs() < 1e-9, "tau must be 0.1");
    }
}
