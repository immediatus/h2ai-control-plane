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
    "Respond with a single JSON object and nothing else:\n",
    "{\"score\": <number 0.0 to 1.0>, \"reason\": \"<one sentence>\"}\n",
    "\n",
    "Score guide:\n",
    "  1.0 — proposal satisfies the criterion\n",
    "  0.5 — proposal partially satisfies the criterion or intent is correct but key detail is missing\n",
    "  0.0 — proposal violates the criterion or does not address it at all\n",
    "\n",
    "Output the JSON object only. No preamble, no explanation outside the JSON."
);

// ── Auditor ───────────────────────────────────────────────────────────────────

/// Auditor approval template. Variables: `{constraints}`, `{proposal}`.
pub const AUDITOR_PROMPT_TEMPLATE: &str = concat!(
    "Review the following proposal for compliance with constraints: {constraints}.\n\n",
    "Proposal:\n{proposal}\n\n",
    "Respond ONLY with JSON: {\"approved\": true, \"reason\": \"<brief explanation>\"}"
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
