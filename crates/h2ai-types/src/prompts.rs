/// Canonical prompt string constants shared across the workspace.
///
/// These constants are the authoritative source for all LLM-facing prompt text.
/// Crates that depend on `h2ai-config` access them via `h2ai_config::prompts`,
/// which re-exports everything here and adds the `PromptTemplate` engine.
///
/// # Variable convention
/// Templates use `{key}` placeholders. Render via `h2ai_config::prompts::PromptTemplate`.

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
pub const EVALUATOR_SYSTEM_PROMPT: &str = "You are a strict evaluator.";

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
