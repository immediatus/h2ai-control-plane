use h2ai_constraints::conflict::ConstraintConflictGraph;
use h2ai_types::events::BranchPrunedEvent;
use std::collections::HashSet;
use std::fmt::Write as FmtWrite;

const CHARS_PER_TOKEN: f64 = 4.0;

/// Compute the per-partial character budget from the synthesis context window.
///
/// Formula: `model_max_tokens × CHARS_PER_TOKEN / (max_k + overhead_factor)`
///
/// `overhead_factor` is the non-partial content budget expressed in "partial slot equivalents"
/// (system context + B1 checklist + Coherence Mandate + synthesis output). The context window
/// is divided into `max_k + overhead_factor` equal slots; partials get `max_k` of them.
///
/// Floors at 32 characters so near-zero budgets (misconfigured tiny models) do not produce
/// empty strings.
#[must_use]
pub fn partial_max_chars(model_max_tokens: u64, max_k: usize, overhead_factor: f64) -> usize {
    let budget = model_max_tokens as f64 * CHARS_PER_TOKEN;
    let per_partial = budget / (max_k as f64 + overhead_factor);
    (per_partial as usize).max(32)
}

/// A pruned proposal that passed at least one binary check.
#[derive(Debug, Clone)]
pub struct PartialPass {
    pub proposal_text: String,
    pub check_results: Vec<(usize, String, bool)>,
    pub score: f64,
}

impl PartialPass {
    pub fn passed_count(&self) -> usize {
        self.check_results.iter().filter(|(_, _, p)| *p).count()
    }

    pub fn passed_check_indices(&self) -> HashSet<usize> {
        self.check_results
            .iter()
            .filter(|(_, _, p)| *p)
            .map(|(i, _, _)| *i)
            .collect()
    }
}

/// Line-safe truncation to `max_chars` characters, snapping to the last newline past the
/// halfway mark when one exists.
pub fn truncate_proposal(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_owned();
    }
    let head: String = text.chars().take(max_chars).collect();
    let cutoff = if let Some(last_newline) = head.rfind('\n') {
        if last_newline > max_chars / 2 {
            last_newline
        } else {
            head.len()
        }
    } else {
        head.len()
    };
    format!(
        "{}\n[... truncated at {} chars; full text omitted to preserve context budget ...]",
        &head[..cutoff],
        char_count,
    )
}

/// Build a `PartialPass` from a `BranchPrunedEvent` given the task's binary checks.
///
/// `offsets` maps each constraint to its slice of the flat `checks` array:
/// each entry is `(constraint_id, start_idx, count)`.
///
/// Per-check attribution uses `ConstraintViolation.check_verdicts` (populated by
/// `parse_check_verdicts` from the LlmJudge CoT output). When a violated constraint
/// has no per-check verdicts, all of its checks are conservatively marked as failed.
/// Checks belonging to unviolated constraints are marked as passing.
/// Checks with no offset entry default to passing (unknown = unviolated).
///
/// Returns `None` when no checks are defined or when the proposal passed zero checks.
pub fn partial_pass_from_event(
    event: &BranchPrunedEvent,
    checks: &[String],
    offsets: &[(String, usize, usize)],
    max_chars: usize,
) -> Option<PartialPass> {
    if checks.is_empty() {
        return None;
    }

    // Build a fast lookup: constraint_id → &ConstraintViolation
    let violated_map: std::collections::HashMap<&str, &h2ai_types::events::ConstraintViolation> =
        event
            .violated_constraints
            .iter()
            .map(|v| (v.constraint_id.as_str(), v))
            .collect();

    let check_results: Vec<(usize, String, bool)> = checks
        .iter()
        .enumerate()
        .map(|(global_idx, check_text)| {
            // Find which constraint owns this global check index.
            let owner = offsets
                .iter()
                .find(|(_, start, count)| global_idx >= *start && global_idx < start + count);

            let passed = match owner {
                None => true, // no constraint mapped → conservative pass
                Some((constraint_id, start, count)) => {
                    let local_idx = global_idx - start;
                    match violated_map.get(constraint_id.as_str()) {
                        None => true, // constraint not violated → check passes
                        Some(v) if v.check_verdicts.is_empty() => {
                            // LLM skipped CHECK N format; infer from fractional score
                            let n_passed = (v.score * *count as f64).round() as usize;
                            local_idx < n_passed
                        }
                        Some(v) => v.check_verdicts.get(local_idx).copied().unwrap_or(false),
                    }
                }
            };
            (global_idx, check_text.clone(), passed)
        })
        .collect();

    let passed_count = check_results.iter().filter(|(_, _, p)| *p).count();
    if passed_count == 0 {
        return None;
    }

    let score = passed_count as f64 / checks.len() as f64;

    Some(PartialPass {
        proposal_text: truncate_proposal(&event.raw_output, max_chars),
        check_results,
        score,
    })
}

/// Greedy set-cover selection of partial-pass proposals.
///
/// `offsets` maps each constraint to its slice of the flat `checks` array:
/// each entry is `(constraint_id, start_idx, count)`.
///
/// Return order is load-bearing: index 0 is the widest-coverage "backbone" — place it
/// first in synthesis prompts to exploit transformer primacy bias. Do not re-sort.
pub fn select_orthogonal_partials(
    all_pruned: &[BranchPrunedEvent],
    checks: &[String],
    offsets: &[(String, usize, usize)],
    max_k: usize,
    max_chars: usize,
) -> Vec<PartialPass> {
    if checks.is_empty() || max_k == 0 {
        return vec![];
    }
    let candidates: Vec<PartialPass> = all_pruned
        .iter()
        .filter_map(|e| partial_pass_from_event(e, checks, offsets, max_chars))
        .filter(|p| p.passed_count() > 0)
        .collect();

    let mut covered: HashSet<usize> = HashSet::new();
    let mut selected: Vec<PartialPass> = Vec::new();
    let mut used: HashSet<usize> = HashSet::new();

    while selected.len() < max_k {
        let best = candidates
            .iter()
            .enumerate()
            .filter(|(idx, _)| !used.contains(idx))
            .max_by(|(_, a), (_, b)| {
                let new_a = a.passed_check_indices().difference(&covered).count();
                let new_b = b.passed_check_indices().difference(&covered).count();
                new_a.cmp(&new_b).then_with(|| {
                    a.score
                        .partial_cmp(&b.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            });
        match best {
            Some((idx, candidate)) => {
                let new_coverage = candidate
                    .passed_check_indices()
                    .difference(&covered)
                    .count();
                if new_coverage == 0 && !covered.is_empty() {
                    break;
                }
                covered.extend(candidate.passed_check_indices());
                used.insert(idx);
                selected.push(candidate.clone());
            }
            None => break,
        }
    }
    selected
}

/// Per-constraint repair target carrying all data needed for the sandwich prompt template.
/// Built by `RetryPolicy::decide()` from aggregated `BranchPrunedEvent` violations.
#[derive(Debug, Clone)]
pub struct RepairTarget {
    pub constraint_id: String,
    /// Natural-language constraint statement (authoritative). From ConstraintDoc.description.
    pub constraint_description: String,
    /// Static YAML remediation hint. Used when verifier_reasons is empty.
    pub remediation_hint: Option<String>,
    /// Pass-criteria text from `criteria.pass` in the constraint YAML.
    /// When Some and non-empty, emitted as TARGET BEHAVIOR block in Slot A.
    pub criteria_pass: Option<String>,
    /// Dynamic verifier reasons from failed proposals, sorted descending by proposal score.
    /// Length grows with retry_count (progressive signal escalation):
    ///   wave 1 → top-1, wave 2 → top-2, wave 3+ → all unique (Jaccard-deduped at 0.7).
    /// Empty when all proposals used static predicates (no LLM verifier reason available).
    pub verifier_reasons: Vec<(f64, String)>,
}

#[derive(Clone, Copy)]
pub struct RepairInput<'a> {
    /// Full text of the best prior proposal across all waves.
    /// Empty string triggers graceful fallback to hint-only format.
    pub prior_proposal_text: &'a str,
    /// Per-constraint repair targets (replaces parallel violated_ids/violated_hints).
    pub targets: &'a [RepairTarget],
    /// Optional zone-3 OSP audit text, appended after REPAIR TARGET blocks when present.
    pub zone3_hints: Option<&'a str>,
    pub conflict_graph: &'a ConstraintConflictGraph,
    pub retry_count: u32,
    pub attempts_remaining: u32,
    pub system_context_with_rubric: &'a str,
    /// Binary check strings B1 checklist injection. Empty = no injection.
    pub checks: &'a [String],
    /// Orthogonally selected partial-pass proposals B2 injection.
    pub partial_passes: &'a [PartialPass],
    /// Best compliance score seen globally across all prior waves.
    /// When provided, emitted as a score-gradient header so the LLM knows how far
    /// the best attempt was from full compliance and can calibrate repair ambition.
    pub prior_best_score: Option<f64>,
    /// semantic correction entries. When non-empty, a DOMAIN KNOWLEDGE CORRECTION
    /// block is prepended to the repair context before any REPAIR TARGET sections.
    /// When empty, behavior is identical to before this field was added.
    pub domain_syntheses: &'a [h2ai_types::gap_i1::DomainSynthesis],
    /// Currently-passing constraints that are coupled to the failing targets via the
    /// conflict graph. Each entry is `(constraint_id, hint_text)`. The hint is the
    /// constraint's `pass_criteria` or `remediation_hint` from the corpus; `None` means
    /// no additional guidance is available. These are injected as a guardrail block so
    /// the LLM cannot silently break them while repairing the failing targets.
    pub coupled_constraint_hints: &'a [(String, Option<String>)],
    /// ALL constraints that passed in the global-best proposal — the complement of the
    /// failing targets. Injected as a "preserve these" block in the repair instructions
    /// so the LLM cannot silently regress previously-satisfied constraints while fixing
    /// the targets. Superset of `coupled_constraint_hints`.
    pub passing_constraint_pins: &'a [(String, Option<String>)],
}

/// Build the CSPR-v2 repair context string.
///
/// Returned string is assigned to `PipelineParams.retry_context` and injected
/// into the next generation wave's system prompt. Anchors the LLM on the best
/// prior proposal and provides targeted per-constraint repair instructions using
/// the three-slot sandwich template (CONSTRAINT REQUIREMENT / VERIFIER INTERPRETATION
/// or GUIDANCE / YOUR TASK). Falls back gracefully when fields are absent.
#[must_use]
pub fn build_repair_context(input: RepairInput<'_>) -> String {
    let RepairInput {
        prior_proposal_text,
        targets,
        zone3_hints,
        conflict_graph,
        retry_count,
        attempts_remaining,
        system_context_with_rubric,
        checks,
        partial_passes,
        prior_best_score,
        domain_syntheses,
        coupled_constraint_hints,
        passing_constraint_pins,
    } = input;

    // Build semantic correction block, prepended before all other content.
    let mut correction_block = String::new();
    if !domain_syntheses.is_empty() {
        for synth in domain_syntheses {
            let source_line = synth
                .source
                .as_deref()
                .map(|s| format!("SOURCE: {s}"))
                .unwrap_or_default();
            let slot = h2ai_config::prompts::I1_SEMANTIC_REPAIR_SLOT
                .replace("{incorrect_pattern}", &synth.incorrect_pattern)
                .replace("{correct_pattern}", &synth.correct_pattern)
                .replace("{mechanistic_reason}", &synth.mechanistic_reason)
                .replace("{source_line}", &source_line);
            correction_block.push_str(&slot);
            correction_block.push('\n');
        }
    }

    let mut out = String::with_capacity(2048);
    if !correction_block.is_empty() {
        writeln!(out, "{correction_block}").unwrap();
    }
    write!(out, "{system_context_with_rubric}").unwrap();

    // B1: compliance checklist at retry >= 1 when binary checks are defined.
    if retry_count >= 1 && !checks.is_empty() {
        let checklist = h2ai_types::prompts::F1_COMPLIANCE_CHECKLIST.replace(
            "{checklist_items}",
            &checks
                .iter()
                .enumerate()
                .map(|(i, c)| format!("{}. {}", i + 1, c))
                .collect::<Vec<_>>()
                .join("\n"),
        );
        write!(out, "\n\n{checklist}").unwrap();
    }

    if prior_proposal_text.is_empty() {
        let feedback = h2ai_types::prompts::CSPR_CONSTRAINT_FEEDBACK_HEADER
            .replace("{retry_count}", &retry_count.to_string())
            .replace("{attempts_remaining}", &attempts_remaining.to_string());
        write!(out, "\n\n{feedback}").unwrap();
    } else {
        // Repair framing comes BEFORE the anchor so the model reads the constraints
        // and preservation rules before it encounters the prior proposal text.
        let score_pct = prior_best_score.map(|s| s * 100.0).unwrap_or(0.0);
        let header = h2ai_types::prompts::CSPR_REPAIR_HEADER
            .replace("{retry_count}", &retry_count.to_string())
            .replace("{score_pct}", &format!("{score_pct:.0}"))
            .replace("{attempts_remaining}", &attempts_remaining.to_string());
        write!(out, "\n\n{header}").unwrap();
        // Emit the full set of passing constraints as a preservation block.
        let pins = if !passing_constraint_pins.is_empty() {
            passing_constraint_pins
        } else {
            coupled_constraint_hints
        };
        if !pins.is_empty() {
            write!(
                out,
                "\n\n{}\n",
                h2ai_types::prompts::CSPR_PASSING_PINS_HEADER
            )
            .unwrap();
            for (pin_id, hint) in pins {
                match hint {
                    Some(h) if !h.is_empty() => {
                        writeln!(out, "  \u{2713} {pin_id}: {h}").unwrap();
                    }
                    _ => {
                        writeln!(out, "  \u{2713} {pin_id}").unwrap();
                    }
                }
            }
        }
        let anchor = h2ai_types::prompts::CSPR_PRIOR_PROPOSAL_BLOCK
            .replace("{retry_count}", &retry_count.to_string())
            .replace("{prior_proposal_text}", prior_proposal_text);
        write!(out, "\n{anchor}").unwrap();
    }

    // Detect conflicting constraint pairs and warn once.
    let violated_ids: Vec<&str> = targets.iter().map(|t| t.constraint_id.as_str()).collect();
    if violated_ids.len() >= 2 {
        'outer: for i in 0..violated_ids.len() {
            for j in (i + 1)..violated_ids.len() {
                let id_a = violated_ids[i];
                let id_b = violated_ids[j];
                if conflict_graph.are_conflicting(id_a, id_b) {
                    write!(
                        out,
                        "\n\n[COMPETING CONSTRAINTS DETECTED: {id_a} and {id_b} have conflicting requirements.\n\
                         Resolution: Fix {id_a} first (hard gate), then verify {id_b} is still satisfied.\n\
                         If both cannot be satisfied simultaneously, satisfy {id_a} and explain why {id_b}\n\
                         cannot be met. Do not attempt to satisfy both by contradiction.]"
                    )
                    .unwrap();
                    break 'outer;
                }
            }
        }
    }

    // Three-slot sandwich template per target.
    for (i, target) in targets.iter().enumerate() {
        let n = i + 1;
        let id = &target.constraint_id;

        if !target.verifier_reasons.is_empty() {
            // Slot A: dynamic verifier reasons, scored and breadth-escalated across waves.
            let (primary_score, primary_reason) = &target.verifier_reasons[0];
            let (target_behavior_block, your_task_text) = match &target.criteria_pass {
                Some(pass) if !pass.is_empty() => (
                    format!("TARGET BEHAVIOR:\n  {}\n\n", pass.trim()),
                    "Produce a new proposal that satisfies the target behavior above.",
                ),
                _ => (
                    String::new(),
                    "Produce a new proposal that satisfies the constraint requirement.",
                ),
            };
            write!(
                out,
                "\n\nREPAIR TARGET {n} — {id}:\n\n\
                CONSTRAINT REQUIREMENT (authoritative):\n  {desc}\n\n\
                VERIFIER INTERPRETATION (best attempt: {pct:.0}% compliance):\n  {primary_reason}\n\n\
                {target_behavior_block}\
                YOUR TASK:\n  {your_task_text}",
                desc = target.constraint_description,
                pct = primary_score * 100.0,
            )
            .unwrap();
            for (score, alt_reason) in target.verifier_reasons.iter().skip(1) {
                write!(
                    out,
                    "\n\n  ALTERNATIVE DIAGNOSIS ({:.0}% attempt): {alt_reason}",
                    score * 100.0,
                )
                .unwrap();
            }
        } else if let Some(ref hint) = target.remediation_hint {
            // Slot B: static YAML hint (contradiction or static predicate).
            write!(
                out,
                "\n\nREPAIR TARGET {n} — {id}:\n\n\
                CONSTRAINT REQUIREMENT (authoritative):\n  {desc}\n\n\
                GUIDANCE:\n  {hint}\n\n\
                YOUR TASK:\n  Produce a new proposal that satisfies the constraint requirement above.",
                desc = target.constraint_description,
            )
            .unwrap();
        } else {
            // Slot C: only constraint description available.
            write!(
                out,
                "\n\nREPAIR TARGET {n} — {id}:\n\n\
                CONSTRAINT REQUIREMENT (authoritative):\n  {desc}\n\n\
                YOUR TASK:\n  Produce a new proposal that satisfies the constraint requirement above.",
                desc = target.constraint_description,
            )
            .unwrap();
        }
    }

    // Coupled constraint guardrail: passing constraints the LLM must not break.
    if !coupled_constraint_hints.is_empty() {
        write!(
            out,
            "\n\n[RELATED CONSTRAINTS THAT MUST NOT BE BROKEN WHILE REPAIRING THE TARGETS ABOVE:"
        )
        .unwrap();
        for (coupled_id, hint) in coupled_constraint_hints {
            match hint {
                Some(h) if !h.is_empty() => {
                    write!(out, "\n  - {coupled_id}: {h}").unwrap();
                }
                _ => {
                    write!(out, "\n  - {coupled_id}: (no additional guidance — ensure this constraint remains satisfied)").unwrap();
                }
            }
        }
        write!(
            out,
            "\nFix the REPAIR TARGET(s) above without violating these related constraints.]"
        )
        .unwrap();
    }

    // Zone-3 OSP audit text appended after all REPAIR TARGET blocks.
    if let Some(hints) = zone3_hints {
        if !hints.is_empty() {
            write!(
                out,
                "\n\n--- OSP AUDIT CONTEXT ---\n{hints}\n--- END OSP AUDIT CONTEXT ---"
            )
            .unwrap();
        }
    }

    // B2: constraint-labeled partial-pass examples.
    if !partial_passes.is_empty() {
        for (n, partial) in partial_passes.iter().enumerate() {
            let status_lines: String = partial
                .check_results
                .iter()
                .map(|(_, check_text, passed)| {
                    if *passed {
                        format!("  ✓ {}  ← SATISFIED (reuse this approach)", check_text)
                    } else {
                        format!("  ✗ {}  ← FAILED (do not repeat this pattern)", check_text)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");

            let block = h2ai_types::prompts::F1_PARTIAL_EXAMPLE
                .replace("{n}", &(n + 1).to_string())
                .replace("{score}", &format!("{:.2}", partial.score))
                .replace("{covered_indices}", &{
                    let mut indices: Vec<usize> =
                        partial.passed_check_indices().into_iter().collect();
                    indices.sort_unstable();
                    indices
                        .iter()
                        .map(|i| (i + 1).to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .replace("{status_lines}", &status_lines)
                .replace("{proposal_text}", &partial.proposal_text);

            write!(out, "\n\n{block}").unwrap();
        }

        write!(
            out,
            "\n\n{}",
            h2ai_types::prompts::F1_PARTIAL_SYNTHESIS_INSTRUCTION
        )
        .unwrap();
    }

    write!(out, "\n\n--- END REPAIR INSTRUCTIONS ---").unwrap();
    out
}

/// Input for the terminal synthesis wave context builder.
pub struct SynthesisInput<'a> {
    /// Orthogonally selected partial-pass proposals, already in coverage order.
    /// Must not be re-sorted. Capped at 3 internally.
    pub partial_passes: &'a [PartialPass],
    pub checks: &'a [String],
    pub system_context_with_rubric: &'a str,
}

/// Build the synthesis wave prompt.
///
/// Combines the compliance checklist (B1), up to 3 coverage-ordered partial examples (B2),
/// and the Coherence Mandate directive. Must only be called when `partial_passes` is non-empty.
#[must_use]
pub fn build_synthesis_context(input: SynthesisInput<'_>) -> String {
    let SynthesisInput {
        partial_passes,
        checks,
        system_context_with_rubric,
    } = input;

    let mut out = String::with_capacity(4096);
    write!(out, "{system_context_with_rubric}").unwrap();
    write!(out, "\n\n{}", h2ai_types::prompts::F1_SYNTHESIS_WAVE_HEADER).unwrap();

    if !checks.is_empty() {
        let checklist = h2ai_types::prompts::F1_COMPLIANCE_CHECKLIST.replace(
            "{checklist_items}",
            &checks
                .iter()
                .enumerate()
                .map(|(i, c)| format!("{}. {}", i + 1, c))
                .collect::<Vec<_>>()
                .join("\n"),
        );
        write!(out, "\n\n{checklist}").unwrap();
    }

    for (n, partial) in partial_passes.iter().take(3).enumerate() {
        let status_lines: String = partial
            .check_results
            .iter()
            .map(|(_, check_text, passed)| {
                if *passed {
                    format!("  ✓ {}  ← SATISFIED (reuse this approach)", check_text)
                } else {
                    format!("  ✗ {}  ← FAILED (do not repeat this pattern)", check_text)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        let block = h2ai_types::prompts::F1_PARTIAL_EXAMPLE
            .replace("{n}", &(n + 1).to_string())
            .replace("{score}", &format!("{:.2}", partial.score))
            .replace("{covered_indices}", &{
                let mut indices: Vec<usize> = partial.passed_check_indices().into_iter().collect();
                indices.sort_unstable();
                indices
                    .iter()
                    .map(|i| (i + 1).to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .replace("{status_lines}", &status_lines)
            .replace("{proposal_text}", &partial.proposal_text);

        write!(out, "\n\n{block}").unwrap();
    }

    write!(out, "\n\n{}", h2ai_types::prompts::F1_SYNTHESIS_FINAL_TASK).unwrap();

    out
}

/// Input for a single sequential constraint graft step.
pub struct GraftInput<'a> {
    /// Current working draft — the seed or result of the previous graft round.
    pub base_text: &'a str,
    /// Full text of the partial proposal that satisfies the target `constraint_ids`.
    /// Passed verbatim to the LLM; no string slicing occurs.
    pub candidate_text: &'a str,
    /// Constraint IDs from the candidate that are missing from the current draft.
    /// These are semantic labels, not byte offsets — safe with any UTF-8 content.
    pub constraint_ids: &'a [String],
    /// System context prepended verbatim (rubric, role, task description).
    pub system_context: &'a str,
}

/// Build a focused graft-step prompt for one round of sequential constraint integration.
///
/// Context size is O(|base| + |candidate|) regardless of total partial count — prevents
/// Lost-in-the-Middle degradation that occurs when all N partials are concatenated in a
/// single synthesis call.
#[must_use]
pub fn build_graft_context(input: &GraftInput<'_>) -> String {
    let GraftInput {
        base_text,
        candidate_text,
        constraint_ids,
        system_context,
    } = input;

    let ids = constraint_ids.join(", ");
    let body = h2ai_types::prompts::H1_GRAFT_CONTEXT
        .replace("{constraint_ids}", &ids)
        .replace("{base_text}", base_text)
        .replace("{candidate_text}", candidate_text);

    format!("{system_context}\n\n{body}")
}

/// Returns constraint IDs where the candidate passes ≥1 check in the constraint's cluster
/// AND the base passes 0 checks in that same cluster.
///
/// `offsets` is `(constraint_id, start_check_index, check_count)` — integer indices into
/// the flat check array. These are NOT byte offsets into string content; no UTF-8 slicing occurs.
///
/// Returns `true` when the candidate is too similar to the base to merit grafting.
///
/// Computes `shared / union` over the passing check indices of `base` and the total
/// check indices of `candidate`. When the ratio exceeds `threshold` (e.g. 0.60) the
/// candidate would contribute minimal new coverage and should be skipped.
pub fn graft_is_redundant(base: &PartialPass, candidate: &PartialPass, threshold: f64) -> bool {
    let base_passing = base.passed_check_indices();
    let candidate_all: HashSet<usize> =
        candidate.check_results.iter().map(|(i, _, _)| *i).collect();
    let union_count = base_passing.union(&candidate_all).count();
    if union_count == 0 {
        return false;
    }
    let shared = base_passing.intersection(&candidate_all).count();
    (shared as f64 / union_count as f64) > threshold
}

/// Returns `true` when all `missing` constraint IDs have already been introduced via
/// a previous graft round (cycle detected in the grafting sequence).
///
/// When the entire `missing` set is a subset of `already_grafted`, no new constraint
/// coverage is possible from this candidate and the round should be skipped.
pub fn grafted_ids_cycle_detected(missing: &[String], already_grafted: &HashSet<String>) -> bool {
    !missing.is_empty() && missing.iter().all(|id| already_grafted.contains(id))
}

/// Returns `true` when the projected token cost of the grafted output would exceed
/// `factor` times the base token estimate.
///
/// Token estimate: `text.len() / 4 + 1` (rough 4-char-per-token proxy). The projection
/// is `(base_text.len() + candidate_text.len()) / 4`; if this exceeds `base_tokens *
/// factor` (e.g. 1.30 = 130%), the combined text would bloat the context window and
/// the candidate should be skipped.
pub fn graft_token_projection_exceeds(base_text: &str, candidate_text: &str, factor: f64) -> bool {
    let base_tokens = (base_text.len() / 4 + 1) as f64;
    let projected = ((base_text.len() + candidate_text.len()) / 4) as f64;
    projected > base_tokens * factor
}

/// Used by the grafting loop to determine which constraint cluster to integrate
/// at each round.
pub fn missing_constraint_ids(
    base: &PartialPass,
    candidate: &PartialPass,
    offsets: &[(String, usize, usize)],
) -> Vec<String> {
    let base_covered = base.passed_check_indices();
    let candidate_covered = candidate.passed_check_indices();

    offsets
        .iter()
        .filter_map(|(constraint_id, start, count)| {
            let mut cluster = *start..*start + *count;
            let base_passes_any = cluster.clone().any(|i| base_covered.contains(&i));
            let candidate_passes_any = cluster.any(|i| candidate_covered.contains(&i));
            if candidate_passes_any && !base_passes_any {
                Some(constraint_id.clone())
            } else {
                None
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// DPPM-MetaRefine support
// ─────────────────────────────────────────────────────────────────────────────

/// Output from one per-cluster parallel solver in the DPPM wave.
#[derive(Debug, Clone)]
pub struct SolverOutput {
    /// Constraint IDs belonging to this cluster.
    pub cluster_ids: Vec<String>,
    /// The proposal text produced by the cluster solver LLM call.
    pub proposal_text: String,
    /// Check indices the seed partial satisfied — prior-wave evidence.
    pub seed_passed_checks: Vec<usize>,
}

/// Detects oscillating MUS pairs from cross-wave pruned events.
/// Returns `(constraint_a, constraint_b)` pairs where A was fixed at wave N+1
/// (disappeared from pruned) but re-appeared at wave N+2, and B was broken
/// at wave N+1 (appeared for the first time). Deduplicates pairs.
pub fn find_oscillation_pairs(
    pruned: &[h2ai_types::events::BranchPrunedEvent],
    _all_check_ids: &[String],
) -> Vec<(String, String)> {
    use std::collections::{BTreeSet, HashSet};

    let all_waves: BTreeSet<u32> = pruned.iter().map(|e| e.retry_count).collect();
    let waves: Vec<u32> = all_waves.into_iter().collect();

    if waves.len() < 3 {
        return vec![];
    }

    let violations_at = |wave: u32| -> HashSet<String> {
        pruned
            .iter()
            .filter(|e| e.retry_count == wave)
            .flat_map(|e| {
                e.violated_constraints
                    .iter()
                    .map(|v| v.constraint_id.clone())
            })
            .collect()
    };

    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut pairs: Vec<(String, String)> = Vec::new();

    for i in 0..waves.len().saturating_sub(2) {
        let w0 = waves[i];
        let w1 = waves[i + 1];
        let w2 = waves[i + 2];

        let v0 = violations_at(w0);
        let v1 = violations_at(w1);
        let v2 = violations_at(w2);

        // C1: violated at w0, NOT at w1, violated again at w2 → oscillates
        let fixed_then_broke: HashSet<String> = v0
            .intersection(&v2)
            .filter(|c| !v1.contains(*c))
            .cloned()
            .collect();

        // C2: appeared at w1 for the first time (not at w0) → the other side of MUS
        let new_at_w1: HashSet<String> = v1.difference(&v0).cloned().collect();

        for c1 in &fixed_then_broke {
            for c2 in &new_at_w1 {
                let pair = if c1 <= c2 {
                    (c1.clone(), c2.clone())
                } else {
                    (c2.clone(), c1.clone())
                };
                if seen.insert(pair.clone()) {
                    pairs.push(pair);
                }
            }
        }
    }

    pairs
}

/// Returns the `PartialPass` (from `partials`) with the maximum number of
/// passing check indices that overlap with `cluster_check_indices`.
/// Returns `None` if no partial has any overlap.
pub fn seed_for_cluster(
    cluster_check_indices: &[usize],
    partials: &[PartialPass],
) -> Option<PartialPass> {
    use std::collections::HashSet;
    let cluster_set: HashSet<usize> = cluster_check_indices.iter().copied().collect();
    partials
        .iter()
        .filter_map(|p| {
            let passed = p.passed_check_indices();
            let overlap = passed.intersection(&cluster_set).count();
            if overlap > 0 {
                Some((overlap, p))
            } else {
                None
            }
        })
        .max_by_key(|(overlap, _)| *overlap)
        .map(|(_, p)| p.clone())
}

// ── Pipeline Resilience: Gap Quality Assessment ───────────────────────────────

/// Verdict from assess_gap_quality. Callers evict with:
/// `matches!(verdict, GapQualityVerdict::Ineffective)`
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GapQualityVerdict {
    /// Not enough post-injection waves yet to judge.
    Pending,
    /// Pass rate improved by at least min_improvement_to_retain.
    Effective,
    /// Pass rate did not improve sufficiently after min_post_injection_waves.
    Ineffective,
}

/// Pure function: classify gap synthesis effectiveness.
///
/// Returns `Pending` when injected_at_wave is None or fewer than
/// cfg.min_post_injection_waves entries in post_injection_pass_rates.
/// Returns `Effective` when latest post-injection pass rate - pre_injection_pass_rate >= cfg.min_improvement_to_retain.
/// Returns `Ineffective` otherwise.
#[must_use]
pub fn assess_gap_quality(
    synthesis: &h2ai_types::gap_i1::DomainSynthesis,
    cfg: &h2ai_config::GapQualityConfig,
) -> GapQualityVerdict {
    if synthesis.injected_at_wave.is_none() {
        return GapQualityVerdict::Pending;
    }
    let post = &synthesis.post_injection_pass_rates;
    if post.len() < cfg.min_post_injection_waves as usize {
        return GapQualityVerdict::Pending;
    }
    let pre = synthesis.pre_injection_pass_rate.unwrap_or(0.0);
    let latest = post.last().copied().unwrap_or(0.0);
    if latest - pre >= cfg.min_improvement_to_retain {
        GapQualityVerdict::Effective
    } else {
        GapQualityVerdict::Ineffective
    }
}

// ── Pipeline Resilience: OOM Guard ───────────────────────────────────────────

/// Error reading /proc/self/status for VmRSS.
#[derive(Debug, thiserror::Error)]
pub enum OomReadError {
    #[error("I/O error reading /proc/self/status: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to parse VmRSS value from /proc/self/status: {0}")]
    Parse(String),
}

/// Signal emitted when RSS exceeds the configured limit.
#[derive(Debug, Clone)]
pub struct OomSignal {
    pub rss_mb: u64,
    pub limit_mb: u64,
}

/// Read VmRSS from /proc/self/status. Returns megabytes.
///
/// On non-Linux platforms always returns `Ok(0)`.
pub fn read_rss_mb() -> Result<u64, OomReadError> {
    #[cfg(target_os = "linux")]
    {
        let contents = std::fs::read_to_string("/proc/self/status")?;
        for line in contents.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb_str = rest.split_whitespace().next().unwrap_or("0");
                let kb: u64 = kb_str
                    .parse()
                    .map_err(|_| OomReadError::Parse(kb_str.to_string()))?;
                return Ok(kb / 1024);
            }
        }
        Err(OomReadError::Parse("VmRSS line not found".to_string()))
    }
    #[cfg(not(target_os = "linux"))]
    {
        Ok(0)
    }
}

/// Pure function: determine if current RSS exceeds the abort threshold.
///
/// Returns `None` when guard is disabled or RSS is below threshold.
/// Returns `Some(OomSignal)` when RSS >= cfg.rss_abort_mb.
#[must_use]
pub fn oom_signal(rss_mb: u64, cfg: &h2ai_config::OomGuardConfig) -> Option<OomSignal> {
    if !cfg.enabled {
        return None;
    }
    if rss_mb >= cfg.rss_abort_mb {
        Some(OomSignal {
            rss_mb,
            limit_mb: cfg.rss_abort_mb,
        })
    } else {
        None
    }
}

/// Builds the full merge-step LLM context by prepending `system_context_with_rubric`
/// (and an optional `balancing_instruction`) to the `INTEGRATION_WAVE_PROMPT` body,
/// with `{constraint_count}` and `{partial_list}` substituted.
///
/// `constraint_checks` maps constraint IDs to their binary check texts.  When
/// non-empty, the exact check wording is embedded inside each cluster section so
/// the integration LLM cannot regress on atomicity primitives or missing Lua steps.
pub fn build_integration_wave_context(
    system_context_with_rubric: &str,
    balancing_instruction: &str,
    solver_outputs: &[SolverOutput],
    constraint_count: usize,
    constraint_checks: &[(String, Vec<String>)],
) -> String {
    let partial_list: String = solver_outputs
        .iter()
        .enumerate()
        .map(|(n, output)| {
            let seed_checks: String = if output.seed_passed_checks.is_empty() {
                "none".to_owned()
            } else {
                output
                    .seed_passed_checks
                    .iter()
                    .map(|i| (i + 1).to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            };

            // Collect the binary check texts for every constraint in this cluster.
            let checks_block: String = output
                .cluster_ids
                .iter()
                .filter_map(|cid| {
                    constraint_checks
                        .iter()
                        .find(|(id, _)| id == cid)
                        .filter(|(_, checks)| !checks.is_empty())
                        .map(|(id, checks)| {
                            let lines: String = checks
                                .iter()
                                .enumerate()
                                .map(|(i, text)| format!("    [{}] {}", i + 1, text))
                                .collect::<Vec<_>>()
                                .join("\n");
                            format!("  {}:\n{}", id, lines)
                        })
                })
                .collect::<Vec<_>>()
                .join("\n");

            if checks_block.is_empty() {
                format!(
                    "### Cluster {}: {}\nPrior evidence (seed solution checks): {}\n---\n{}\n---",
                    n + 1,
                    output.cluster_ids.join(", "),
                    seed_checks,
                    output.proposal_text
                )
            } else {
                format!(
                    "### Cluster {}: {}\nRequired binary checks (ALL must pass in unified proposal):\n{}\nPrior evidence (seed solution checks): {}\n---\n{}\n---",
                    n + 1,
                    output.cluster_ids.join(", "),
                    checks_block,
                    seed_checks,
                    output.proposal_text
                )
            }
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let body = h2ai_types::prompts::INTEGRATION_WAVE_PROMPT
        .replace("{constraint_count}", &constraint_count.to_string())
        .replace("{partial_list}", &partial_list);

    let mut out = String::with_capacity(4096);
    out.push_str(system_context_with_rubric);
    if !balancing_instruction.is_empty() {
        out.push_str("\n\n");
        out.push_str(balancing_instruction);
    }
    out.push_str("\n\n");
    out.push_str(&body);
    out
}
