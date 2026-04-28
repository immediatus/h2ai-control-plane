# H2AI Control Plane — Differentiation

This document explains what H2AI Control Plane does that existing multi-agent frameworks do not, why those differences matter, and where the tradeoffs lie.

---

## The Comparison Landscape

The major multi-agent LLM frameworks as of 2026:

| Framework | Primary model | Language | State model | Scale target |
|---|---|---|---|---|
| **LangGraph** | Graph-based workflow DAG | Python | In-process / Redis | Single process, Python ecosystem |
| **AutoGen** | Conversational agent loop | Python | In-process thread storage | Research, prototyping |
| **CrewAI** | Role-based task delegation | Python | In-process | Small teams, quick setup |
| **Semantic Kernel** | Planner + plugin orchestration | C# / Python | In-process | Enterprise Microsoft stack |
| **H2AI Control Plane** | Physics-bounded agent swarm | Rust | Event-sourced CRDT on NATS | Production, auditable, multi-node |

---

## What H2AI Does Differently

### 1. Coordination Cost Is Measured, Not Guessed

**Every other framework:** You set `max_agents` (or equivalent). The number comes from intuition, documentation examples, or trial and error. There is no principled basis for why 3 agents is better than 5 for a given task.

**H2AI:** `N_max` is derived from measured parameters. α (serial bottleneck) and β_eff (pairwise reconciliation cost modulated by agent divergence) are calibrated before any task runs. The system computes the exact N at which adding another agent starts degrading output quality and enforces it. The calibration can be re-run when the adapter pool changes, the task domain shifts, or models are updated.

**Why it matters:** Multi-agent systems exhibit retrograde behavior — beyond a critical N, more agents produce worse results than fewer. This has been confirmed empirically for LLM ensembles across multiple 2023–2025 studies. A framework that lets you add agents without bound is not neutral; it is actively harmful once the coordination ceiling is crossed. H2AI tells you where that ceiling is.

### 2. System State Is an Immutable Event Log

**Every other framework:** State lives in the LLM context window (lossy, non-reproducible, lost on restart) or in-process memory (lost on crash). If the orchestrator restarts mid-task, work is lost and the task starts over.

**H2AI:** Every state transition — task bootstrapped, agent dispatched, proposal received, proposal rejected, merge decision made — is appended to an immutable NATS JetStream event log. Crash recovery is replay from offset 0. The complete provenance chain of every output is preserved. The same state model runs in development (Local Plan) and production (Cloud Plan / Kubernetes).

**Why it matters in practice:**
- **Audit:** Every rejected proposal is a permanent record with the reason, the violated constraint, and the remediation hint. For regulated industries (SOX, HIPAA, SOC2), this is not optional.
- **Reproducibility:** The exact sequence of events that produced an output is replayable. You can reproduce the state of any task at any point in its execution.
- **Debugging:** "What did agent A output before the merge?" is a JetStream query, not a log grep.

### 3. Constraints Are Typed Predicates, Not Prompt Instructions

**Every other framework:** Safety and compliance constraints are encoded in the system prompt as natural language. "Do not use G1GC." "Always use idempotency keys." These are suggestions that the LLM may or may not follow, with no verification mechanism.

**H2AI:** Constraints are `ConstraintDoc` instances with typed `ConstraintPredicate` bodies:
- `VocabularyPresence { required_keywords, threshold }` — output must contain these terms
- `NegativeKeyword { prohibited_terms }` — output must not contain these terms
- `RegexMatch { pattern }` — deterministic structural check
- `NumericThreshold { field, min, max }` — numeric value bounds
- `LlmJudge { prompt_template, threshold }` — semantic evaluation by a judge model
- `Composite { operator: And|Or, predicates }` — boolean combinations

Each predicate has a severity: `Hard` (gate — violation rejects the proposal), `Soft` (weighted penalty), or `Advisory` (logged but not blocking). `constraint_error_cost = 1 − compliance` feeds back into the BFT merge strategy selector.

**Why it matters:** Prompt instructions are probabilistic. A typed constraint predicate with a Hard gate is deterministic. A proposal that violates ADR-004 ("use Redis Lua for atomic budget operations") is *structurally impossible to reach the human* — the Auditor rejects it before merge. The J_eff gate prevents generation from starting if the task manifest doesn't cover the constraint vocabulary. This is defense-in-depth, not a single-layer probabilistic guardrail.

### 4. Tool-Using Agent Risk Is Quantified

**Every other framework:** Tools are function signatures or capability declarations. A file-writing agent and a pure-reasoning agent are treated identically by the orchestration layer. The risk of incorrect tool use is not modeled.

**H2AI:** `AgentTool` flags (`Shell`, `WebSearch`, `CodeExecution`, `FileSystem`) directly affect three measured quantities:

| Capability | α contribution | β_base contribution | c_i (error cost) |
|---|---|---|---|
| Pure LLM | ~0 | ~0 | 0.1–0.3 |
| WebSearch | +0.01–0.02 | +0.005 | 0.2–0.4 |
| FileSystem | +0.02–0.05 | +0.010 | 0.4–0.6 |
| CodeExecution | +0.03–0.08 | +0.015 | 0.5–0.7 |
| Shell | +0.05–0.15 | +0.020 | 0.6–0.9 |

When `max(c_i)` exceeds the BFT threshold (0.95), `MergeStrategy` switches from score-ordered to Krum — Byzantine-resistant selection that minimizes the damage from a single high-error-cost agent producing a wrong output with irreversible side effects. This is automatic; no configuration is required.

**Why it matters:** An incorrect file-write cannot be undone. An incorrect shell command may be catastrophic. Treating these agents the same as pure-reasoning agents (as all other frameworks do) is unsafe. H2AI quantifies the error cost and adjusts both the merge strategy and the NATS credential scope accordingly.

### 5. Agent Credentials Are Task-Scoped

**Every other framework:** Long-lived API keys or shared credentials in containers. An agent that runs for task A still holds credentials for task B.

**H2AI:** Each task gets a fresh NATS NKey scoped to exactly the subjects that task's agents can publish to. The NKey is provisioned at dispatch time and expires when the task closes. Scoping is enforced at the NATS server — not by application code.

**Why it matters:** In a tool-using system, an agent that can access the orchestration event bus can tamper with other agents' outputs. An agent that retains credentials after task completion can impersonate future agents. Scoped, expiring credentials eliminate both attack surfaces. This is a security property that no Python-based framework provides.

### 6. Human Decision Is O(1), Not O(N)

**Every other framework:** The human reviews N agent outputs, identifies contradictions, selects the best, and feeds corrections back into the loop. For N=5 agents with 3 surviving proposals, this is O(N) reading and O(N²) comparison work.

**H2AI:** The CRDT Merge Authority presents:
- The merged diff grouped by target component (not raw agent outputs)
- The tombstone panel: every rejected proposal with reason and constraint cost
- The autonomic shift timeline: every MAPE-K intervention
- The physics panel: live κ_eff, J_eff, N_max, current MergeStrategy

The human makes **one decision** on the merged diff. Contradictions have already been resolved by the merge strategy (ScoreOrdered, ConsensusMedian, or Krum). The work is O(1) regardless of N.

---

## What H2AI Does NOT Do Better

**Setup speed:** CrewAI and LangGraph have a 5-minute path to a working prototype. H2AI requires configuring a NATS instance, a constraint corpus, and an adapter pool before the first task runs. This is appropriate for production deployments; it is not appropriate for quick experimentation.

**Python ecosystem integration:** The primary runtime is Rust. Python tools, LangChain integrations, and Python-native LLM libraries require adapter wrappers. For teams whose toolchain is entirely Python, H2AI adds integration friction.

**Visual workflow design:** LangGraph Studio provides a GUI for designing agent workflows. H2AI's topologies are selected automatically from calibration data — the tradeoff is that you get less manual control over the exact graph shape in exchange for physics-grounded automatic selection.

**Breadth of agent patterns:** AutoGen and CrewAI have large catalogs of pre-built agent roles, conversation patterns, and task decomposition strategies. H2AI has three topologies (Ensemble+CRDT, HierarchicalTree, TeamSwarmHybrid) and a role system (Coordinator, Executor, Evaluator, Synthesizer). The focus is depth (correctness and auditability) over breadth (variety of patterns).

---

## When to Choose H2AI

H2AI is the right choice when:

1. **Output correctness is auditable:** You need a permanent, queryable record of every agent output, every rejected proposal, and every merge decision. Regulated industries, legal review, financial decisions.

2. **Tool-using agents write state:** Agents have file, shell, or external API access. Wrong outputs have irreversible consequences. The BFT merge strategy and scoped credentials matter.

3. **The constraint space is explicit:** You have architectural decision records, compliance requirements, or operational constraints that should *block* non-compliant proposals, not just discourage them.

4. **Scale beyond single-process:** You need crash recovery, multi-node orchestration, or horizontal scaling. The event-sourced model on NATS JetStream supports all three without architectural changes.

5. **Ensemble size needs to be principled:** You are running enough agents that the coordination ceiling is a real concern, or you need to justify your ensemble size to stakeholders beyond "we tried 3 and it worked."

H2AI is **not** the right choice when:

- You need a working prototype in 30 minutes.
- All agents are pure-reasoning (no tools), runs are stateless, and you don't need audit.
- Your team is fully Python-native and the integration cost exceeds the value.
- The task is simple enough that N=1 agent is sufficient and ensembles add no value.
