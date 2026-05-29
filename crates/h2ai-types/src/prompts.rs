//! Canonical prompt string constants shared across the workspace.
//!
//! These constants are the authoritative source for all LLM-facing prompt text.
//! Crates that depend on `h2ai-config` access them via `h2ai_config::prompts`,
//! which re-exports everything here and adds the `PromptTemplate` engine.
//!
//! Templates use `{key}` placeholders. Render via `h2ai_config::prompts::PromptTemplate`.

// ── Verification / evaluation ─────────────────────────────────────────────────

/// G-Eval–style chain-of-thought rubric (arxiv 2303.16634).
/// No substitution variables. `verification.rs` appends `\n\nProposal:\n{output}`.
pub const COT_RUBRIC: &str = concat!(
    "Evaluate the following proposal against these criteria. ",
    "For each criterion, state whether the proposal satisfies it (yes/partial/no) and why. ",
    "Then output a JSON object: {\"score\": 0.0_to_1.0, \"reason\": \"one sentence\"}\n\n",
    "Criteria:\n",
    "1. Does the proposal directly address the stated task?\n",
    "2. Is the response accurate and free of factual errors?\n",
    "3. Are all required constraints satisfied?\n",
    "4. Is the response appropriately concise (not padded with unnecessary content)."
);

/// System prompt for the LLM evaluator role.
///
/// Owns the response contract: JSON schema, score scale, and output format.
/// Constraint rubrics (## Semantic Rules) contain only behavioral criteria —
/// they must NOT repeat the JSON format instruction; the framework injects it here.
pub const EVALUATOR_SYSTEM_PROMPT: &str = concat!(
    "You are an architectural compliance evaluator.\n",
    "\n",
    "You will receive a compliance criterion (what to check) followed by a proposal to evaluate.\n",
    "\n",
    "--- Standard scoring (no binary checks) ---\n",
    "Respond with a single JSON object: {\"score\": <0.0 to 1.0>, \"reason\": \"<one sentence>\"}\n",
    "  1.0 — proposal satisfies the criterion\n",
    "  0.5 — proposal partially satisfies or intent correct but key detail missing\n",
    "  0.0 — proposal violates the criterion or does not address it\n",
    "\n",
    "--- Anchored CoT scoring (when 'Binary compliance checks' section is present) ---\n",
    "Evaluate each numbered check and write: CHECK N: <text> → PRESENT or MISSING\n",
    "Then compute: score = count(PRESENT) / total_checks\n",
    "Then output the JSON: {\"score\": <computed value>, \"reason\": \"<comma-separated verdicts>\"}\n",
    "\n",
    "Always end your response with the JSON object on its own line."
);

/// Adversarial variant of the evaluator system prompt.
///
/// Used when explorer slot configs carry `rejection_criteria` — each explorer was already
/// instructed to look for a specific failure mode, so the verifier should probe adversarially
/// rather than check rubric compliance. This partially restores verifier independence
/// without requiring a separate model family.
/// System prompt for binary YES/NO factual checks (`SemanticPresence`, `SemanticExclusion`, `SemanticOrdering`).
/// Must NOT use the adversarial evaluator framing — factual presence checks require neutral classification,
/// not hostile scoring. The adversarial framing causes the model to answer NO regardless of content.
pub const BINARY_CLASSIFIER_SYSTEM_PROMPT: &str =
    "You are a precise text classifier. Answer with exactly one word: YES or NO. \
     Do not add punctuation, explanation, or any other text.";

pub const ADVERSARIAL_EVALUATOR_SYSTEM_PROMPT: &str = concat!(
    "You are a hostile reviewer. Your goal is NOT to check whether this proposal follows the rubric.\n",
    "Your goal is to find the single most likely way this proposal fails silently, violates a \n",
    "constraint under realistic conditions, or produces incorrect results in production.\n",
    "\n",
    "Process:\n",
    "1. Identify the most suspicious claim in the proposal — the one most likely to be wrong.\n",
    "2. Check each criterion adversarially: assume the proposal is trying to deceive you.\n",
    "3. Score based on how much doubt you can eliminate, not how much compliance you observe.\n",
    "\n",
    "--- Standard scoring (no binary checks) ---\n",
    "Respond with a single JSON object: {\"score\": <0.0 to 1.0>, \"reason\": \"<one sentence>\"}\n",
    "  1.0 — you actively tried to break this proposal and could not find a failure\n",
    "  0.5 — you found a plausible failure but the proposal might survive it\n",
    "  0.0 — you found a concrete way this proposal fails\n",
    "\n",
    "--- Anchored CoT scoring (when 'Binary compliance checks' section is present) ---\n",
    "For each check, assume the proposal fails it until proven otherwise.\n",
    "Write: CHECK N: <text> → PRESENT or MISSING (with your adversarial reasoning)\n",
    "Then compute: score = count(PRESENT) / total_checks\n",
    "Then output the JSON: {\"score\": <computed value>, \"reason\": \"<comma-separated verdicts>\"}\n",
    "\n",
    "Always end your response with the JSON object on its own line."
);

// ── Auditor ───────────────────────────────────────────────────────────────────

/// System prompt for the auditor role.
///
/// Establishes the adversarial post-incident engineer persona and mandates JSON output.
/// Kept separate from the user-level `AUDITOR_PROMPT_TEMPLATE` so the role framing
/// is not overridden by the explorer's constraint-corpus system context.
///
/// # Design rationale — single-family independence
///
/// Naive confirmatory prompts ("is this compliant?") produce strong approval bias:
/// the same model that generated the proposal tends to affirm it (Constitutional AI,
/// Bai et al. 2022; sycophancy studies on self-evaluation). Two techniques from the
/// literature restore independence without a separate model family:
///
/// 1. **Null-hypothesis reversal** (adversarial framing): treat the proposal as
///    non-compliant by default; require explicit per-constraint evidence of compliance
///    before approving. This mirrors the critique step in Constitutional AI and the
///    adversarial scoring in MT-Bench / LLM-as-Judge (Zheng et al. 2023).
///
/// 2. **Anchored chain-of-thought** (Self-Refine, Madaan et al. 2023): force the
///    model to enumerate each constraint and state a verdict before reaching the
///    overall decision. Structured `CoT` dramatically reduces confirmation bias compared
///    to a single yes/no question (Wei et al. 2022).
pub const AUDITOR_SYSTEM_PROMPT: &str = concat!(
    "You are a post-incident engineer. You have been called in after production failures ",
    "caused by exactly the class of constraints listed below being violated.\n",
    "Your job is to find whether a proposal contains the same class of bug — ",
    "NOT to confirm it looks fine.\n\n",
    "Treat each proposal as guilty until proven innocent. ",
    "For each constraint, hunt for the specific way it is violated. ",
    "Only record SATISFIES when you find explicit, concrete evidence in the proposal text.\n\n",
    "You MUST end your response with a JSON object on its own line — no exceptions:\n",
    "{\"approved\": <true only if ALL constraints SATISFY>, ",
    "\"violated\": [\"<constraint-id>\", ...], ",
    "\"reason\": \"<one line per violated constraint>\"}"
);

/// `serde(default)` helper for `AuditorConfig::system_prompt`.
#[must_use]
pub fn auditor_system_prompt_default() -> String {
    AUDITOR_SYSTEM_PROMPT.into()
}

/// Adversarial auditor user-message template. Variables: `{constraints}`, `{proposal}`.
///
/// Sent as the user turn; the system turn uses `AUDITOR_SYSTEM_PROMPT`.
/// The full constraint predicate text lives in the system context compiled by
/// `compiler.rs`; `{constraints}` here is the list of IDs used as anchors.
pub const AUDITOR_PROMPT_TEMPLATE: &str = concat!(
    "Constraints to audit: {constraints}\n\n",
    "For each constraint ID:\n",
    "  1. What does this constraint prohibit or require?\n",
    "  2. Does the proposal contain the prohibited pattern or omit the required one?\n",
    "  3. Verdict: SATISFIES or VIOLATES — cite the exact line or phrase.\n\n",
    "Proposal:\n{proposal}\n\n",
    "After your per-constraint analysis, output JSON on its own line:\n",
    "{{\"approved\": <true only if every constraint is SATISFIES>, ",
    "\"violated\": [\"<constraint-id>\", ...], ",
    "\"reason\": \"<one line per violated constraint>\"}}"
);

// ── TAO retry loop ────────────────────────────────────────────────────────────

/// Emitted as TAO observation when the turn passes all checks.
pub const TAO_OBSERVATION_PASS: &str = "verification passed";

/// Emitted as TAO observation when the `verify_pattern` regex fails. Variable: `{turn}`.
pub const TAO_OBSERVATION_FAIL_PATTERN: &str = "pattern not matched on turn {turn}; retrying";

/// Emitted as TAO observation when JSON schema validation fails. Variables: `{turn}`, `{error}`.
pub const TAO_OBSERVATION_FAIL_SCHEMA: &str =
    "schema validation failed on turn {turn}: {error}; retrying";

/// Instruction appended to the task on TAO retry. Variable: `{turn}`.
pub const TAO_RETRY_INSTRUCTION: &str = concat!(
    "[OBSERVATION turn {turn}]: output did not satisfy verification. ",
    "Revise your response."
);

// ── Synthesis phase ───────────────────────────────────────────────────────────

/// Stage 1 prompt — asks the LLM to critique all proposals and return a `CritiqueDocument` JSON.
/// Variables: `{task_description}`, `{constraint_list}`, `{proposals_block}`, `{critique_schema}`.
pub const SYNTHESIS_CRITIQUE_PROMPT: &str = concat!(
    "You are a critical reviewer. Analyse the proposals below for the given task and constraints.\n\n",
    "Task:\n{task_description}\n\n",
    "Constraints:\n{constraint_list}\n\n",
    "Proposals:\n{proposals_block}\n\n",
    "Produce a JSON critique document matching this schema exactly:\n{critique_schema}\n\n",
    "Rules:\n",
    "- List every proposal in proposal_critiques.\n",
    "- Identify all contradictions between proposals in contradictions.\n",
    "- End with synthesis_guidance: a single paragraph instructing the synthesiser ",
    "which strengths to incorporate, which weaknesses to avoid, and how to resolve each contradiction.\n",
    "Return ONLY the JSON object. No prose before or after."
);

/// Stage 2 prompt — asks the LLM to write the final synthesised output from the critique.
/// Variables: `{task_description}`, `{constraint_list}`, `{proposals_block}`, `{critique_document}`.
pub const SYNTHESIS_WRITE_PROMPT: &str = concat!(
    "You are a synthesis writer. Using the critique document below, ",
    "produce a single unified response to the task that incorporates identified strengths, ",
    "avoids identified weaknesses, and resolves all contradictions as directed.\n\n",
    "Task:\n{task_description}\n\n",
    "Constraints:\n{constraint_list}\n\n",
    "Original proposals:\n{proposals_block}\n\n",
    "Critique document:\n{critique_document}\n\n",
    "Write only the final synthesised response. No preamble, no meta-commentary."
);

// ── Constraint-Informed Synthesis prompts ─────────────────────────────

/// B1 mechanism: compliance checklist block injected into the repair context at retry ≥ 1.
/// Variables: `{checklist_items}` — newline-joined numbered list produced by
/// `build_checklist_items(checks)` in repair.rs.
pub const F1_COMPLIANCE_CHECKLIST: &str = concat!(
    "--- COMPLIANCE CHECKLIST (satisfy ALL of the following) ---\n",
    "{checklist_items}\n",
    "Your response MUST satisfy every item above. ",
    "Check each one explicitly before finalising.\n",
    "--- END CHECKLIST ---"
);

/// B2 mechanism: one partial-pass example block.
/// Variables: `{n}` index (1-based), `{score}` (e.g. "0.67"), `{covered_indices}` (e.g. "1, 3"),
/// `{status_lines}` — newline-joined ✓/✗ lines, `{proposal_text}` — truncated proposal.
pub const F1_PARTIAL_EXAMPLE: &str = concat!(
    "--- PARTIAL EXAMPLE {n} (score={score}, covers checks: {covered_indices}) ---\n",
    "The following proposal satisfies SOME but not ALL constraints.\n",
    "Use it as a source of correct techniques — do NOT copy its failures.\n\n",
    "Per-constraint status:\n",
    "{status_lines}\n\n",
    "Proposal text (may be truncated):\n",
    "{proposal_text}\n",
    "--- END PARTIAL EXAMPLE {n} ---"
);

/// B2 mechanism: synthesis instruction appended after partial examples when > 0 partials present.
pub const F1_PARTIAL_SYNTHESIS_INSTRUCTION: &str = concat!(
    "SYNTHESIS TASK: Produce a single proposal that combines the SATISFIED approaches\n",
    "from the examples above, while fixing all FAILED items. Do not introduce any\n",
    "pattern not already justified above."
);

/// A mechanism: synthesis wave wrapper injected around the partial examples and checklist.
/// Variables: `{checklist_block}` (F1_COMPLIANCE_CHECKLIST rendered),
/// `{example_blocks}` (F1_PARTIAL_EXAMPLE blocks joined), `{final_task}` (F1_SYNTHESIS_FINAL_TASK).
pub const F1_SYNTHESIS_WAVE_HEADER: &str = concat!(
    "--- SYNTHESIS WAVE ---\n",
    "Previous attempts partially satisfied the constraints. ",
    "Your task is to produce a single final proposal combining the best elements ",
    "of the examples below.\n\n",
    "Examples are ordered by constraint coverage breadth (most checks covered first), ",
    "not by raw score. Example 1 is the structural backbone — preserve its architecture.\n"
);

/// A mechanism: final task directive with Coherence Mandate.
pub const F1_SYNTHESIS_FINAL_TASK: &str = concat!(
    "FINAL SYNTHESIS TASK:\n",
    "Produce ONE proposal that satisfies EVERY item in the checklist above.\n\n",
    "COHERENCE MANDATE (strictly enforced):\n",
    "  - You must NOT simply concatenate solutions from the examples above.\n",
    "  - If the partial proposals use architecturally incompatible foundations\n",
    "    (e.g., one uses Postgres polling, another uses Redis pub-sub), you must\n",
    "    unify them under a SINGLE coherent architecture before addressing\n",
    "    any individual constraint.\n",
    "  - Do NOT output contradictory state mechanisms. A design that stores state\n",
    "    in two mutually exclusive systems simultaneously is invalid, regardless of\n",
    "    whether it passes individual binary checks in isolation.\n",
    "  - Your output must be a unified, internally consistent engineering design.\n",
    "    Verify architectural coherence before verifying constraint checklist items.\n",
    "--- END SYNTHESIS WAVE ---"
);

/// Single graft-step prompt. Variables:
/// `{constraint_ids}` — comma-joined constraint IDs being integrated,
/// `{base_text}` — current working draft (seed or result of previous graft),
/// `{candidate_text}` — full text of the partial that satisfies the target constraints.
pub const H1_GRAFT_CONTEXT: &str = concat!(
    "--- GRAFT STEP (constraints: {constraint_ids}) ---\n",
    "You are extending a working draft to satisfy additional constraints.\n\n",
    "CURRENT DRAFT:\n",
    "{base_text}\n\n",
    "CONSTRAINT CLUSTER TO INTEGRATE ({constraint_ids}):\n",
    "The following proposal satisfies the above constraints. ",
    "Extract ONLY the techniques relevant to those constraints — do not import other patterns.\n\n",
    "{candidate_text}\n\n",
    "TASK: Extend the current draft to satisfy the constraints listed above.\n",
    "Rules:\n",
    "  1. Do NOT violate constraints already satisfied in the current draft.\n",
    "  2. Do NOT copy architectural patterns from the candidate unless required by the constraints.\n",
    "  3. If a constraint is architecturally incompatible with the current draft, prefer the\n",
    "     current draft's architecture and document the tradeoff explicitly.\n",
    "  4. Output the complete updated draft, not a diff.\n",
    "--- END GRAFT STEP ---"
);
