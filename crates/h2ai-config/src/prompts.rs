/// Lightweight prompt template engine and all workspace-level prompt constants.
///
/// # Template syntax
/// Variables are written as `{key}`. Use `PromptTemplate::render` to substitute them.
/// Literal braces that are NOT variables (e.g., inside a JSON example) must not collide
/// with `{key}` form — bare `{` / `}` without an alphanumeric key are left as-is.
///
/// # Single source of truth
/// - Types in `h2ai-types` that embed prompt defaults (TaoConfig, VerificationConfig,
///   AuditorConfig) reference `h2ai_types::prompts::*`.
/// - Everything from that module is re-exported here so callers have one import path:
///   `h2ai_config::prompts`.
/// - Planner prompts live here because h2ai-planner (and orchestrator) depend on h2ai-config.
// Re-export all base constants from h2ai-types so callers only import h2ai-config::prompts.
pub use h2ai_types::prompts::{
    AUDITOR_PROMPT_TEMPLATE, COT_RUBRIC, EVALUATOR_SYSTEM_PROMPT, TAO_OBSERVATION_FAIL_PATTERN,
    TAO_OBSERVATION_FAIL_SCHEMA, TAO_OBSERVATION_PASS, TAO_RETRY_INSTRUCTION,
};

// ── Template engine ───────────────────────────────────────────────────────────

/// A `&'static str` prompt template with `{key}` variable substitution.
#[derive(Debug, Clone, Copy)]
pub struct PromptTemplate(pub &'static str);

impl PromptTemplate {
    /// Substitute every `{key}` occurrence using `vars` pairs.
    /// Returns an owned `String` with all matched placeholders replaced.
    pub fn render(&self, vars: &[(&str, &str)]) -> String {
        vars.iter().fold(self.0.to_owned(), |s, (key, val)| {
            s.replace(&format!("{{{key}}}"), val)
        })
    }

    /// Return the raw template string without any substitution.
    pub fn as_str(&self) -> &'static str {
        self.0
    }
}

impl std::fmt::Display for PromptTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

// ── Planner — task decomposer ─────────────────────────────────────────────────

/// System prompt for the decomposer LLM call. No variables.
pub const DECOMPOSER_SYSTEM: PromptTemplate =
    PromptTemplate("You are a senior software architect. Respond only with valid JSON.");

/// Task prompt for decomposing a manifest into a subtask plan.
/// Variables: `{description}`, `{constraints}`.
pub const DECOMPOSER_TASK: PromptTemplate = PromptTemplate(concat!(
    "You are decomposing a complex task into an ordered subtask plan.\n",
    "\n",
    "Original task: {description}\n",
    "Constraints: {constraints}\n",
    "\n",
    "Decompose this into 2 to 7 subtasks. Each subtask must be a specific, ",
    "independently executable step whose output is useful to later subtasks.\n",
    "\n",
    "Respond ONLY with valid JSON matching this schema exactly:\n",
    "{\n",
    "  \"subtasks\": [\n",
    "    {\n",
    "      \"description\": \"<specific instruction for this subtask>\",\n",
    "      \"depends_on\": [<0-based indices of prior subtasks this depends on>],\n",
    "      \"role_hint\": \"<Executor|Evaluator|Synthesizer|Coordinator|null>\"\n",
    "    }\n",
    "  ]\n",
    "}"
));

// ── Planner — plan reviewer ───────────────────────────────────────────────────

/// System prompt for the plan-reviewer LLM call. No variables.
pub const PLAN_REVIEWER_SYSTEM: PromptTemplate =
    PromptTemplate("You are a critical plan reviewer. Respond only with valid JSON.");

/// Task prompt for reviewing a proposed subtask decomposition.
/// Variables: `{original_description}`, `{subtask_summary}`.
pub const PLAN_REVIEWER_TASK: PromptTemplate = PromptTemplate(concat!(
    "You are reviewing a subtask decomposition plan.\n",
    "\n",
    "Original task: {original_description}\n",
    "\n",
    "Proposed plan:\n{subtask_summary}\n",
    "\n",
    "Evaluate:\n",
    "1. Does this plan fully address the original task with no obvious missing steps?\n",
    "2. Is the dependency order logical?\n",
    "\n",
    "Respond ONLY with valid JSON:\n",
    "{\"approved\": true, \"reason\": \"...\"}"
));
