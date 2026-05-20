use h2ai_constraints::conflict::ConstraintConflictGraph;

pub struct RepairInput<'a> {
    /// Full text of the best prior proposal across all waves.
    /// Empty string triggers graceful fallback to hint-only format.
    pub prior_proposal_text: &'a str,
    /// Constraint IDs that failed in the last wave.
    pub violated_ids: &'a [String],
    /// Remediation hint per violated constraint (parallel to violated_ids).
    pub violated_hints: &'a [Option<String>],
    pub conflict_graph: &'a ConstraintConflictGraph,
    pub retry_count: u32,
    pub attempts_remaining: u32,
    pub system_context_with_rubric: &'a str,
}

/// Build the CSPR-v2 repair context string.
///
/// Returned string is assigned to `PipelineParams.retry_context` and injected
/// into the next generation wave's system prompt. Anchors the LLM on the best
/// prior proposal and provides targeted per-constraint repair instructions.
/// Falls back to hint-only format when `prior_proposal_text` is empty.
pub fn build_repair_context(input: RepairInput<'_>) -> String {
    let RepairInput {
        prior_proposal_text,
        violated_ids,
        violated_hints,
        conflict_graph,
        retry_count,
        attempts_remaining,
        system_context_with_rubric,
    } = input;

    let mut out = String::with_capacity(2048);
    out.push_str(system_context_with_rubric);

    if !prior_proposal_text.is_empty() {
        out.push_str(&format!(
            "\n\n--- PRIOR PROPOSAL (wave {retry_count}, use as repair anchor) ---\n\
            {prior_proposal_text}\n\
            --- END PRIOR PROPOSAL ---"
        ));
    }

    let header = if prior_proposal_text.is_empty() {
        format!(
            "\n\n--- CONSTRAINT FEEDBACK (iteration {retry_count}) ---\n\
            The following constraints were violated. Fix ALL of these in your next response:\n\n\
            {attempts_remaining} retry attempt(s) remaining."
        )
    } else {
        format!(
            "\n\n--- CONSTRAINT REPAIR INSTRUCTIONS (iteration {retry_count}) ---\n\
            The proposal above violates the following constraints. Apply TARGETED repairs only.\n\
            Do NOT change sections that comply with other constraints.\n\
            {attempts_remaining} attempt(s) remaining."
        )
    };
    out.push_str(&header);

    if violated_ids.len() >= 2 {
        'outer: for i in 0..violated_ids.len() {
            for j in (i + 1)..violated_ids.len() {
                let id_a = &violated_ids[i];
                let id_b = &violated_ids[j];
                if conflict_graph.are_conflicting(id_a, id_b) {
                    out.push_str(&format!(
                        "\n\n[COMPETING CONSTRAINTS DETECTED: {id_a} and {id_b} have conflicting requirements.\n\
                         Resolution: Fix {id_a} first (hard gate), then verify {id_b} is still satisfied.\n\
                         If both cannot be satisfied simultaneously, satisfy {id_a} and explain why {id_b}\n\
                         cannot be met. Do not attempt to satisfy both by contradiction.]"
                    ));
                    break 'outer;
                }
            }
        }
    }

    for (i, id) in violated_ids.iter().enumerate() {
        let hint = violated_hints
            .get(i)
            .and_then(|h| h.as_deref())
            .unwrap_or("Ensure the constraint condition is satisfied.");
        out.push_str(&format!("\n\nREPAIR TARGET {} — {id}:\n{}", i + 1, hint));
    }

    out.push_str("\n\n--- END REPAIR INSTRUCTIONS ---");
    out
}
