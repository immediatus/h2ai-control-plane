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

// ── Epistemic committee decomposition — 3-step pipeline ──────────────────────
//
// Prompt text lives here; builder functions in h2ai-orchestrator call render()
// for dynamic substitution.

/// Step 1 system: failure mode analyst. No variables.
pub const DECOMPOSITION_STEP1_SYSTEM: PromptTemplate = PromptTemplate(
    "You are a failure mode analyst. Your job is to read constraint requirements and \
identify the specific requirements that general-purpose engineers miss on first pass — \
not the obvious ones, but the ones that cause production incidents.",
);

/// Step 2 system: persona designer. No variables.
pub const DECOMPOSITION_STEP2_SYSTEM: PromptTemplate = PromptTemplate(
    "You are designing expert reviewer personas for a technical committee. Each persona \
must be defined by what they notice FIRST when reading a proposal — anchored to \
specific professional experience with a concrete failure type, not a generic title.",
);

/// Step 3 system: JSON formatter. No variables.
pub const DECOMPOSITION_STEP3_SYSTEM: PromptTemplate = PromptTemplate(
    "You are a JSON formatter. Convert structured expert role descriptions into a precise \
JSON array. Output only valid JSON — no markdown fences, no explanation.",
);

/// Step 1 task: identify what engineers miss per constraint domain.
/// Variables: `{thinking_context}`, `{description}`, `{constraints}`.
pub const DECOMPOSITION_STEP1_TASK: PromptTemplate = PromptTemplate(concat!(
    "{thinking_context}",
    "TASK: {description}\n",
    "\n",
    "ACTIVE CONSTRAINTS:\n{constraints}\n",
    "\n",
    "For each constraint domain above, answer three questions:\n",
    "1. What is the single most counter-intuitive requirement ",
    "(the one a general-purpose engineer misses on first pass)?\n",
    "2. What is the typical violation pattern — how does the design usually fail this?\n",
    "3. What epistemic blindspot causes engineers to miss it ",
    "(wrong mental model, missing context, false assumption)?"
));

/// Step 2 task: design one expert persona per constraint domain.
/// Variables: `{analysis}`, `{n_total}`, `{domain_assignments}`, `{integration_idx}`.
pub const DECOMPOSITION_STEP2_TASK: PromptTemplate = PromptTemplate(concat!(
    "FAILURE MODE ANALYSIS:\n{analysis}\n",
    "\n",
    "Design exactly {n_total} expert reviewer personas — one per domain below plus one integration role.\n",
    "\n",
    "REQUIRED DOMAIN ASSIGNMENTS (do not add, skip, or merge):\n",
    "{domain_assignments}",
    "  Role {integration_idx}: integration reviewer — detects cascade failures when ALL of the above ",
    "domains fail simultaneously.\n",
    "\n",
    "For EACH role, use the failure mode from the analysis for that domain. Write:\n",
    "- role_frame: \"You are a [role] who has [specific experience with this domain's failure].\"\n",
    "  The role must change what the expert notices FIRST — anchored to that failure, not a generic title.\n",
    "- reasoning_style: Choose backward_chaining (trace from the failure backward), ",
    "devil_s_advocate (prove the design is wrong), ",
    "first_principles (derive from invariants), or step_by_step (enumerate state transitions).\n",
    "- what_they_hunt: The specific failure this expert looks for first.\n",
    "\n",
    "{n_total} roles total. One per domain. Describe in plain text."
));

/// Step 3 task: convert role descriptions to a JSON committee array.
/// Variables: `{roles}`, `{n_max}`, `{corpus_domains}`.
pub const DECOMPOSITION_STEP3_TASK: PromptTemplate = PromptTemplate(concat!(
    "EXPERT ROLES:\n{roles}\n",
    "\n",
    "Convert these roles into a JSON array. Maximum {n_max} elements.\n",
    "Each element must have exactly these fields:\n",
    "- \"role_frame\": string. 1-2 sentences starting with \"You are a [specific role].\"\n",
    "- \"cot_style\": exactly one of: \"step_by_step\", \"devil_s_advocate\", ",
    "\"first_principles\", \"backward_chaining\", \"none\"\n",
    "- \"focus_mandate\": string. The constraint domain(s) this expert covers.\n",
    "- \"rejection_criteria\": string. The specific failure mode this expert hunts.\n",
    "- \"constraint_domains\": JSON array of strings. Choose ONLY from the valid vocabulary ",
    "below — copy strings VERBATIM, do not paraphrase or invent new names. ",
    "Empty array [] when this expert covers none of the listed domains.\n",
    "  Valid vocabulary: {corpus_domains}\n",
    "- \"search_enabled\": boolean. true ONLY when this role requires current external knowledge ",
    "(latest CVEs, library versions, regulations). false for most roles.\n",
    "\n",
    "Output ONLY the JSON array. No markdown, no explanation."
));

// ── Complexity probe ──────────────────────────────────────────────────────────

/// Prefix injected into the system context for probe-mode LLM calls.
pub const PROBE_SYSTEM_PREFIX: &str = "[PROBE_MODE: structure assessment only]";

/// Task prompt for complexity probe calls. No variables.
pub const PROBE_TASK: PromptTemplate =
    PromptTemplate("Briefly outline your approach to this task. Focus on which constraints apply.");

// ── Verification ──────────────────────────────────────────────────────────────

/// Task prompt for LlmJudge evaluation calls.
/// Variables: `{rubric}`, `{output}`.
pub const VERIFICATION_TASK: PromptTemplate =
    PromptTemplate("Criterion:\n{rubric}\n\nProposal:\n{output}");

// ── SRANI grounding ───────────────────────────────────────────────────────────

/// System prompt for the LLM researcher grounder (tier-0 SRANI escalation).
/// No variables.
pub const SRANI_RESEARCHER_SYSTEM: PromptTemplate =
    PromptTemplate("You are a technical grounding advisor. Respond with valid JSON only.");

/// Task prompt for the LLM researcher grounder.
/// Variables: `{fabricated}`, `{task_description}`.
pub const SRANI_RESEARCHER_TASK: PromptTemplate = PromptTemplate(concat!(
    "These components were introduced but are NOT in the specification: {fabricated}.\n",
    "Task context: {task_description}\n",
    "Provide spec-compliant alternatives. Respond with JSON: ",
    r#"{"alternatives": ["..."], "statement": "..."}"#,
));

/// System prompt for the web-search distillation step.
/// Instructs the LLM to compress raw search results into concise factual prose.
/// No variables.
pub const SRANI_DISTILL_SYSTEM: PromptTemplate = PromptTemplate(
    "You are a technical fact extractor. \
     Given web search results, extract only the key factual technical statements \
     relevant to the task. Return 2-4 concise sentences. \
     No URLs, no headings, no lists — plain prose only.",
);

/// Task prompt for the web-search distillation step.
/// Variables: `{task_description}`, `{raw_results}`.
pub const SRANI_DISTILL_TASK: PromptTemplate = PromptTemplate(concat!(
    "Task: {task_description}\n\n",
    "Search results:\n{raw_results}\n\n",
    "Extract the most relevant technical facts for this task in 2-4 sentences.",
));

// ── Thinking Loop ─────────────────────────────────────────────────────────────

/// System prompt for archetype selection LLM call. Always uses Capable tier.
pub const THINKING_ARCHETYPE_SYSTEM: &str =
    "You are a cognitive strategist selecting expert reviewer archetypes for a technical problem. \
     Each archetype must be defined by a specific professional lens that will surface insights \
     a generic reviewer would miss. Output only a valid JSON array — no markdown, no explanation.";

/// Archetype selection task for iteration 1 (no prior thinking context).
/// Variables: `{description}`, `{constraints}`, `{research_context}`, `{n}`.
pub const THINKING_ARCHETYPE_SELECT_ITER1: PromptTemplate = PromptTemplate(concat!(
    "TASK: {description}\n\n",
    "ACTIVE CONSTRAINTS: {constraints}\n\n",
    "RESEARCH CONTEXT: {research_context}\n\n",
    "Select exactly {n} expert archetypes. Each archetype must have a fundamentally different \
     cognitive lens — not just different titles, but different things they notice FIRST.\n\n",
    "Output a JSON array where each element has:\n",
    "- \"name\": kebab-case identifier\n",
    "- \"persona\": 2-3 sentences starting with \"You are a [role] who...\"\n",
    "- \"scope\": the specific slice of the problem this archetype reasons about\n",
    "- \"confidence\": 0.0–1.0 (how confident this archetype is in their domain)\n",
    "- \"tau\": 0.0–1.0 (reasoning temperature; lower for precision roles, higher for creative)\n",
    "- \"model_tier\": \"fast\", \"standard\", or \"capable\"\n",
    "- \"cot_style\": \"step_by_step\", \"devil_s_advocate\", \"first_principles\", \
     \"backward_chaining\", or \"none\"\n\n",
    "Output ONLY the JSON array."
));

/// Archetype selection task for iteration N>1. Feeds only distilled summary, not raw outputs
/// (Think Twice principle: discard intermediates to force independent re-evaluation).
/// Variables: `{description}`, `{understanding}`, `{tensions}`, `{n}`.
pub const THINKING_ARCHETYPE_SELECT_ITERN: PromptTemplate = PromptTemplate(concat!(
    "TASK: {description}\n\n",
    "PRIOR SYNTHESIS:\n{understanding}\n\n",
    "UNRESOLVED TENSIONS (these gaps were NOT resolved in the previous iteration):\n{tensions}\n\n",
    "Select exactly {n} archetypes. REPLACE any archetype whose perspective did not contribute \
     to resolving any tension above. New archetypes must specifically target the unresolved gaps.\n\n",
    "Output a JSON array with the same fields as before: \
     name, persona, scope, confidence, tau, model_tier, cot_style.\n\n",
    "Output ONLY the JSON array."
));

/// Task prompt for per-archetype brainstorm session.
/// Variables: `{description}`, `{research_context}`, `{cot_instruction}`.
pub const THINKING_BRAINSTORM_TASK: PromptTemplate = PromptTemplate(concat!(
    "{cot_instruction}\n\n",
    "TASK: {description}\n\n",
    "RESEARCH CONTEXT: {research_context}\n\n",
    "Working strictly within your assigned scope:\n\n",
    "PROBLEM ANALYSIS:\n",
    "Identify the 3-5 most critical sub-problems, risks, and key decisions from your perspective. \
     Focus on what a general-purpose engineer would miss.\n\n",
    "SOLUTION SKETCH:\n",
    "Outline your recommended approach. Be concrete — name specific mechanisms, thresholds, \
     and failure modes. End with: {\"confidence\": <0.0-1.0>}"
));

/// System prompt for synthesis LLM call. Always uses Capable tier (stage-level routing).
pub const THINKING_SYNTHESIS_SYSTEM: &str =
    "You are a synthesis facilitator merging insights from multiple expert perspectives. \
     Weight higher-confidence views more heavily when resolving conflicts (ReConcile method). \
     Output a single JSON object — no markdown, no explanation.";

/// Synthesis task: confidence-weighted merge of all archetype outputs.
/// Variables: `{perspectives}`, `{prior_understanding}`.
pub const THINKING_SYNTHESIS_TASK: PromptTemplate = PromptTemplate(concat!(
    "PRIOR UNDERSTANDING (from previous iteration — empty on first pass):\n{prior_understanding}\n\n",
    "ARCHETYPE PERSPECTIVES (weight by confidence score):\n{perspectives}\n\n",
    "Synthesise these perspectives into a unified understanding. Higher-confidence views \
     should dominate when archetypes conflict.\n\n",
    "Output a JSON object:\n",
    r#"{"shared_understanding": "...", "tensions": ["...", "..."], "coverage_score": 0.0}"#, "\n\n",
    "- shared_understanding: 3-5 sentences capturing what all archetypes agree on\n",
    "- tensions: list of specific unresolved conflicts between archetype views\n",
    "- coverage_score: 0.0–1.0 self-assessment of how completely the problem space is covered\n\n",
    "Output ONLY the JSON object."
));

/// System prompt for LLM quality gate call. Always uses Capable tier.
pub const THINKING_QUALITY_GATE_SYSTEM: &str =
    "You are a readiness evaluator deciding whether a problem analysis is complete enough \
     to begin generating solutions. Answer with exactly YES or NO followed by one sentence.";

/// Quality gate task.
/// Variables: `{understanding}`, `{tensions}`, `{coverage}`.
pub const THINKING_QUALITY_GATE_TASK: PromptTemplate = PromptTemplate(concat!(
    "SYNTHESIS:\n{understanding}\n\n",
    "UNRESOLVED TENSIONS:\n{tensions}\n\n",
    "COVERAGE SCORE: {coverage}\n\n",
    "Are all critical problem dimensions resolved enough to begin generating solutions? \
     Answer YES or NO with one sentence reason."
));

// ── Explorer system context — constraint entry templates ─────────────────────
//
// These templates are used by h2ai-context::compiler to build the system context
// injected into every explorer prompt. Keeping them here ensures prompt text
// is configurable and visible alongside other workspace prompts.

/// Task manifest section header. Variables: `{manifest}`.
pub const COMPILER_TASK_MANIFEST: PromptTemplate = PromptTemplate("## Task Manifest\n{manifest}");

/// Entry for a Hard or include_rubric=true LlmJudge constraint.
/// Variables: `{id}`, `{rubric}`.
pub const COMPILER_CONSTRAINT_HARD_RUBRIC: PromptTemplate =
    PromptTemplate("## {id} Constraint\n{rubric}");

/// Entry for a Soft LlmJudge constraint when rubric is withheld.
/// Variables: `{id}`.
pub const COMPILER_CONSTRAINT_ACTIVE_ID: PromptTemplate =
    PromptTemplate("## Active Constraint: {id}");

/// Vocabulary constraint block. Variables: `{id}`, `{terms}`.
pub const COMPILER_CONSTRAINT_VOCABULARY: PromptTemplate =
    PromptTemplate("## {id} Constraints\n{terms}");

/// Guidance suffix appended to any constraint entry that has a remediation hint.
pub const COMPILER_CONSTRAINT_GUIDANCE_SUFFIX: &str = "\nGuidance: ";

/// Requirement suffix used for soft-LlmJudge constraints (rubric withheld, hint shown).
pub const COMPILER_CONSTRAINT_REQUIREMENT_SUFFIX: &str = "\nRequirement: ";

/// Ordering constraint entry. Variables: `{id}`, `{first}`, `{then}`.
pub const COMPILER_CONSTRAINT_ORDERING: PromptTemplate = PromptTemplate(concat!(
    "## Active Constraint: {id} [ordering requirement]\n",
    "Required sequence: '{first}' must occur before '{then}' in your proposal."
));

/// Semantic presence constraint entry. Variables: `{id}`, `{concept}`.
pub const COMPILER_CONSTRAINT_PRESENCE: PromptTemplate = PromptTemplate(concat!(
    "## Active Constraint: {id} [semantic requirement]\n",
    "Your proposal must address: {concept}."
));

/// Semantic exclusion constraint entry. Variables: `{id}`, `{pattern}`.
pub const COMPILER_CONSTRAINT_EXCLUSION: PromptTemplate = PromptTemplate(concat!(
    "## Active Constraint: {id} [exclusion requirement]\n",
    "Your proposal must NOT include: {pattern}."
));

/// Ordering sub-requirement appended inside a Composite constraint entry.
/// Variables: `{first}`, `{then}`.
pub const COMPILER_COMPOSITE_ORDERING_DETAIL: PromptTemplate =
    PromptTemplate("\nRequired sequence: '{first}' must occur before '{then}'.");

/// Presence sub-requirement appended inside a Composite constraint entry.
/// Variables: `{concept}`.
pub const COMPILER_COMPOSITE_PRESENCE_DETAIL: PromptTemplate =
    PromptTemplate("\nRequired: {concept}.");

/// Exclusion sub-requirement appended inside a Composite constraint entry.
/// Variables: `{pattern}`.
pub const COMPILER_COMPOSITE_EXCLUSION_DETAIL: PromptTemplate =
    PromptTemplate("\nMust NOT include: {pattern}.");

/// Decomposition step-1 constraint block entry.
/// Variables: `{id}`, `{domains}`, `{rubric}`, `{hint}`.
pub const DECOMPOSITION_CONSTRAINT_ENTRY: PromptTemplate = PromptTemplate(concat!(
    "CONSTRAINT {id} [{domains}]\n",
    "Rubric: {rubric}\n",
    "Remediation hint: {hint}"
));

#[cfg(test)]
mod thinking_prompts_tests {
    use super::*;

    #[test]
    fn semantic_ordering_template_renders_all_fields() {
        let rendered = COMPILER_CONSTRAINT_ORDERING.render(&[
            ("id", "C-005"),
            ("first", "account debit"),
            ("then", "Kafka publish"),
        ]);
        assert!(rendered.contains("C-005"));
        assert!(rendered.contains("account debit"));
        assert!(rendered.contains("Kafka publish"));
    }

    #[test]
    fn archetype_select_iter1_renders_required_fields() {
        let rendered = THINKING_ARCHETYPE_SELECT_ITER1.render(&[
            ("description", "design a caching layer"),
            ("constraints", "CONSTRAINT-001"),
            ("research_context", "Redis is the standard choice"),
            ("n", "3"),
        ]);
        assert!(rendered.contains("design a caching layer"));
        assert!(rendered.contains("CONSTRAINT-001"));
        assert!(rendered.contains("Redis is the standard choice"));
        assert!(rendered.contains("3"));
    }

    #[test]
    fn synthesis_task_renders_perspectives_and_prior() {
        let rendered = THINKING_SYNTHESIS_TASK.render(&[
            ("perspectives", "arch A: use Redis"),
            ("prior_understanding", ""),
        ]);
        assert!(rendered.contains("arch A: use Redis"));
    }
}
