# From Theory to Implementation

This guide bridges the mathematical apparatus (`docs/architecture/math-apparatus.md`) and the H2AI runtime. It answers: given a real engineering task, how do you configure the system to extract the maximum value from multi-agent collaboration without paying unnecessary coordination cost?

The guide is organised as: topology selection protocol → topology catalog → team-scale configuration → dark knowledge management → implementation mapping.

---

## The Four-Question Topology Protocol

Before submitting any task to H2AI, four questions locate the optimal topology. Each question has a single concrete test:

### Question 0 — Is this a team task or a solo task?

**Test:** Does more than one person need to contribute, review, or approve the final output?

- **Yes (multiple humans):** Use the **Team-Swarm Hybrid**. Skip to the Team Configuration section. Designate a swarm liaison, structure the agent swarm with role-differentiated temperatures, set a review gate for the highest error-cost agent.
- **No (single human coordinator):** Continue to Question 1.

### Question 1 — What is the error cost?

**Test:** How damaging is a hallucination that reaches production undetected?

- **High** (security, compliance, financial, production migrations, irreversible changes): Prioritise containment. Use low-τ agents (τ ≤ 0.2), enforce consensus within subtrees, require human merge authority.
- **Low** (brainstorming, drafts, exploratory analysis, documentation): Relax containment. Allow high-τ agents, human as loose filter.

### Question 2 — Does diversity of approach have value?

**Test:** Is the best answer obvious in advance, or do multiple distinct strategies need to be compared?

- **Yes** (architecture decisions, security approach selection, root-cause analysis, test strategy design): Use Ensemble or Hierarchical tree. Preserve all agent contributions to the merge point. Do not let any intermediate node resolve disagreements before the human sees them.
- **No** (data formatting, deterministic transformation, single-answer lookup, known API signature): Use Oracle or Star. Diversity adds noise, not signal.

### Question 3 — How many agents does the task require?

**Test:** Compare against the AI N_max ceiling of approximately 6.

- `N ≤ 3`: Oracle or Ensemble both viable. Prefer Ensemble if error cost is medium or higher.
- `3 < N ≤ 6` (at or below AI N_max): **Ensemble + CRDT**. This is the default frontier topology for solo H2AI work.
- `N > 6`: **Hierarchical Tree**. Each subtree must stay within its own N_max. A human or trusted low-τ coordinator agent at each internal node.

### Decision flowchart

```
Task arrives
    │
    ▼
Q0: Multiple humans? ──Yes──► Team-Swarm Hybrid
    │
    No
    │
    ▼
Q1: Error cost high? ──No──► Oracle or Star (low-stakes)
    │
    Yes
    │
    ▼
Q2: Diversity has value? ──No──► Low-τ consensus topology
    │
    Yes
    │
    ▼
Q3: N ≤ 6? ──Yes──► Ensemble + CRDT (default frontier)
    │
    No
    │
    ▼
    Hierarchical Tree
```

---

## Topology Catalog

### Topology 1 — Oracle (single agent)

One AI agent produces output directly consumed by the human.

**Pareto scores:** T = 50%, E = 88%, D = 20%

| Axis | Score | Mechanism |
|------|-------|-----------|
| Throughput | Medium | Serial — bounded by single-agent capacity |
| Containment | High | No propagation path; human sees one output |
| Diversity | Low | Single τ produces a single epistemic stance |

**Failure mode:** No redundancy check on hallucination. A Byzantine fault propagates directly to the human with no intermediate quarantine.

**Use when:** Single deterministic right answer, speed matters more than diversity — debugging a specific error message, formatting structured data, generating a unit test for a known function signature.

---

### Topology 2 — Flat Panel (N agents, no merge structure)

Multiple AI agents produce independent outputs. The human reads all outputs without a defined merge protocol.

**Pareto scores:** T = 18%, E = 18%, D = 90%

| Axis | Score | Mechanism |
|------|-------|-----------|
| Throughput | Low (large N) | Human attention is the contention bottleneck; α spikes past N ≈ 4 |
| Containment | Low | All outputs reach the human unfiltered |
| Diversity | High | All epistemic stances preserved |

**Failure mode:** Human attention becomes the serial bottleneck Amdahl predicted. The panel degrades into noise past a small N.

**Use when:** Early-stage exploration where the full range of approaches is more valuable than any individual answer — technology selection surveys, "what approaches exist for X" brainstorming. Not appropriate when any single hallucination would waste significant human time.

---

### Topology 3 — Star (human hub, AI spokes)

The human acts as active coordinator, routing sub-tasks to specialized agents and receiving outputs on a per-task basis.

**Pareto scores:** T = 52%, E = 78%, D = 75%

| Axis | Score | Mechanism |
|------|-------|-----------|
| Throughput | Medium | Human hub processes one spoke at a time; serial coordination cost |
| Containment | High | Every output passes through the hub before integration |
| Diversity | High | Human observes all specialist outputs before merging |

**Failure mode:** Hub bottleneck. At N > 4–5 spokes, the human coordinator's context-switching cost becomes the dominant α term.

**Use when:** Tasks with clearly separable sub-domains and a human who has enough context to route correctly — a code review where each file type maps to a specialist agent, a research task where each subtopic maps to a different retrieval strategy.

---

### Topology 4 — Pipeline (sequential chain)

Agents form a directed chain. Each agent transforms the output of the previous agent.

**Pareto scores:** T = 48%, E = 18%, D = 20%

| Axis | Score | Mechanism |
|------|-------|-----------|
| Throughput | Medium | High when dependency structure is genuine and each step is verifiable |
| Containment | Low | Errors cascade; each agent treats the previous agent's hallucination as authoritative |
| Diversity | Low | Each step filters and narrows toward a single answer |

**Failure mode:** Error compounding. A hallucination at step 1 is revised at step 2, reformatted at step 3, and delivered with high polish and low accuracy.

**Use when:** Tasks with a strict sequential dependency where each step's output is independently verifiable — draft, fact-check, format, then human sign-off. Each intermediate verification gate must be explicit; without it, the pipeline is a hallucination amplifier.

**Status: Dominated.** No other topology scores worse on all three axes simultaneously. Avoid for high-stakes work.

---

### Topology 5 — Ensemble + CRDT Merge ⭐

Multiple parallel AI agents produce divergent outputs. The human performs an explicit CRDT merge: preserving useful contributions, discarding hallucinations, combining the epistemic diversity into a coherent artifact.

**Pareto scores:** T = 84%, E = 84%, D = 90% — **Pareto non-dominated**

| Axis | Score | Mechanism |
|------|-------|-----------|
| Throughput | High | Agents run in parallel, each below the AI N_max ceiling |
| Containment | High | Human merge step quarantines Byzantine faults before they propagate |
| Diversity | High | All agent contributions survive to the merge point |

**This is the only single-human topology that scores high on all three axes simultaneously.**

**Failure mode:** Human merge quality degrades if the human lacks the domain knowledge to distinguish a hallucination from an unconventional-but-correct output. The CRDT merge is only as good as the merge authority's contextual judgement.

**Use when:** Architecture proposals, security reviews, test generation with multiple strategies, any task where diversity of approach has value and the human has the domain knowledge to judge outputs. **The default topology for high-value H2AI work with a single human coordinator.**

---

### Topology 6 — Hierarchical Tree ⭐

For N that exceeds the single-human merge authority's capacity, the tree extends to multiple levels. AI leaf agents produce outputs, intermediate merge authorities (AI coordinators or human team leads) perform sub-merges, and the human principal performs the root merge.

**Pareto scores:** T = 96%, E = 96%, D = 60% — **Pareto non-dominated**

| Axis | Score | Mechanism |
|------|-------|-----------|
| Throughput | Very high | Coordinator converts O(N²) edges to O(N) edges |
| Containment | Very high | Multi-level quarantine; faults propagate to at most k peers |
| Diversity | Medium | Intermediate merge steps may filter minority views |

**Failure mode:** Diversity collapse at intermediate levels. If sub-merge authorities apply consensus rather than CRDT semantics, the root receives pre-filtered output where interesting outliers have been discarded. Fix: instruct intermediate merge authorities to preserve dissenting views as annotated items, not to resolve them.

**Use when:** Large-N tasks exceeding any individual human's attention bandwidth — comprehensive codebase audits, multi-domain research synthesis, large-scale test generation campaigns.

---

### Topology 7 — Team-Swarm Hybrid ⭐

The topology that governs most real H2AI work. A human team with its own coordination hierarchy operates alongside a specialized agent swarm, connected through a single liaison node.

**Pareto scores:** T = 84%, E = 91%, D = 95% — **Pareto non-dominated**

| Axis | Score | Mechanism |
|------|-------|-----------|
| Throughput | High | Parallel swarm + human team operate concurrently within their ceilings |
| Containment | Very high | Intra-swarm review gates + liaison as interface quarantine + human principal as final merge authority |
| Diversity | Very high | Swarm contributes temperature diversity; human team contributes experiential diversity; both survive to the principal's merge |

**The Pareto frontier topology for any sustained H2AI collaboration at team scale.**

---

## Pareto Summary

| Topology | T | E | D | Status |
|----------|---|---|---|--------|
| Hierarchical Tree | 96% | 96% | 60% | **Frontier** |
| Team-Swarm Hybrid | 84% | 91% | 95% | **Frontier** |
| Ensemble + CRDT | 84% | 84% | 90% | **Frontier** |
| Star | 52% | 78% | 75% | Dominated |
| Oracle | 50% | 88% | 20% | Dominated |
| Pipeline | 48% | 18% | 20% | Dominated |
| Flat Panel | 18% | 18% | 90% | Dominated |

**Reading the table:** A topology is Pareto non-dominated if no alternative beats it on all three axes at once. The three frontier topologies sit at the top — each scores high across most cells, with one deliberate trade-off (Hierarchical Tree sacrifices some D; Team-Swarm Hybrid sacrifices some T; Ensemble + CRDT is the low-overhead default when neither extreme matters).

---

## Team-Scale Configuration

### Designating the Swarm Liaison

The liaison is the single interface node between the human team and the agent swarm. In Amdahl's Law terms, the liaison is the serial fraction α of the entire system: every task requiring human judgement must pass through this node, and no amount of swarm parallelism can bypass it.

The maximum system speedup is bounded by `1 / α_liaison` regardless of swarm size.

**Selection criteria:** The liaison should be the team member with the highest joint CG across both:
1. The human team (they understand implicit team constraints — "we don't store sessions" knowledge)
2. The agent swarm (they have invested in context compilation — `J_eff > 0.6` with the swarm coordinator)

This is not necessarily the most senior engineer. It is the engineer who spans the technical-organisational boundary — in practice, often the tech lead who already performs a merge-authority role within the human team.

**When the liaison is saturated:** Do not add more liaisons — that creates two coordination problems where there was one. Either raise `CG(H_liaison, SC)` through better context compilation and coordinator calibration, or split the swarm into separate sub-swarms each with their own liaison (applying the Hierarchical Tree pattern at the team level).

### Intra-Swarm Role Specialisation

Not all agents in the swarm should be interchangeable. H2AI uses abstract topological roles (`AgentRole`) — the *domain* the role applies to is encoded in `system_context`, not in the role enum. Different roles warrant different τ calibrations and different error cost weights:

| AgentRole | τ default | c_i default | Function | Edge constraint |
|-----------|-----------|-------------|----------|-----------------|
| `Coordinator` | 0.05 | 0.1 | Routes sub-tasks to other Explorers, summarises for liaison | Low τ: must be deterministic and auditable |
| `Executor` | 0.40 | 0.5 | Produces primary output artifacts | Subject to review gates declared in `review_gates[]` |
| `Evaluator` | 0.10 | 0.9 | Constraint and quality checking | **Review gate**: blocks Executor output if check fails; high c_i triggers BFT merge path |
| `Synthesizer` | 0.80 | 0.1 | Combines, summarises, or documents other outputs | High τ: diversity has value; errors easily corrected |
| `Custom` | (required) | (required) | Any domain-specific role not covered above | τ and c_i must be explicitly supplied |

The role differentiation is not bureaucracy — it is temperature-calibrated error containment. An `Evaluator` at `c_i = 0.9` with a review gate converts a Byzantine fault from a propagation-factor-of-N event to a propagation-factor-of-1 event. The `Coordinator` at τ ≈ 0 decouples deterministic routing from the higher-τ exploration of `Executor` nodes.

### The Three N_max Ceilings

A Team-Swarm Hybrid has three simultaneous ceilings that must all hold:

```
N_max^human-team = sqrt((1 − α_H) · CG_bar_HH / κ_base^H)  ≈ 10

N_max^swarm      = sqrt((1 − α_A) · CG_bar_AA / κ_base^A)  ≈ 6

N_max^interface  = sqrt((1 − α_liaison) · CG(H_liaison, SC) / κ_base^liaison)
```

The interface ceiling — the number of concurrent swarm tasks the liaison can effectively coordinate — is typically 3–5. **This is the binding constraint in most team-swarm deployments**, not the intra-swarm or intra-human ceilings.

---

## Dark Knowledge Management

### Why ADRs Increase J_eff

The Dark Knowledge Gap (`J_eff`) measures the Jaccard overlap between the knowledge in a task submission and the knowledge the task actually requires. When `J_eff < 0.4`, the system returns `ContextUnderflowError` — the Auditor has insufficient constraint coverage to reject bad proposals.

ADRs convert tacit team knowledge into explicit, machine-readable constraint statements:

- Phrases like "must not", "is forbidden", "never" → Auditor prohibition statements
- Phrases like "must", "is required", "always" → Auditor requirement statements
- Service names, component names, compliance references → Scope identifiers

**The `## Constraints` section is the only part of an ADR that the compiler uses for rejection decisions.** Write it as a bullet list of hard rules in imperative language. Every bullet is a potential `BranchPrunedEvent` reason.

### The J_eff Effect in Practice

**Without ADRs:** A task about budget enforcement returns `ContextUnderflowError` — `J_eff = 0.12`, below the 0.4 threshold. The system refuses to proceed.

**With ADRs:** The same task returns `202 Accepted` — `J_eff = 0.71`. Three Explorers generate proposals. One proposes reading budget from a cache (faster, but stale). The Auditor catches it — "ADR-004: budget checks must read from Redis atomic counters, never from cache" — and publishes `BranchPrunedEvent`. Two valid proposals reach the Merge Authority.

### Minimum Viable ADR Corpus

A corpus with fewer than 5 ADRs typically produces `J_eff < 0.4` for real architectural tasks. Target:

| Requirement | Minimum |
|-------------|---------|
| Total ADRs | 5 |
| Strong `## Constraints` sections | All of them |
| ADRs citing compliance references (SOX, OWASP, etc.) | ≥ 1 |
| ADRs covering the critical path (latency, consistency) | ≥ 2 |
| ADRs covering error handling (retries, idempotency) | ≥ 1 |

Diagnose low `J_eff` by examining the `missing_coverage` field in the `ContextUnderflowError` response — it lists the task requirement categories with no matching ADR constraint.

---

## Implementation Mapping

### How H2AI Implements Each Theorem

| Mathematical concept | H2AI implementation |
|---------------------|---------------------|
| Calibrate α, κ_base, CG_mean | `POST /calibrate` → `CalibrationCompletedEvent` |
| J_eff gate (Def 10) | Dark Knowledge Compiler in `crates/h2ai-context`; `ContextUnderflowError` if below 0.4 |
| N_max ceiling (Prop 1) | Explorer count capped at N_max during `TopologyProvisionedEvent` |
| Multiplication condition (Prop 3) | Phase 2.5 hard gate; `MultiplicationConditionFailedEvent` names which condition failed |
| θ_coord threshold (Prop 2) | Stored in calibration cache; enforced at topology construction |
| CRDT merge (Prop 4, 5) | `MergeStrategy::CrdtSemilattice` in `crates/h2ai-state` |
| BFT merge (Prop 5 safety constraint) | `MergeStrategy::BftConsensus` when `max(c_i) > 0.85`; `ConsensusRequiredEvent` signals this path |
| Auditor constraint checking (Def 10) | `BranchPrunedEvent` with ADR citation, emitted by Auditor adapter |
| MAPE-K retry on zero survival | `ZeroSurvivalEvent` → `crates/h2ai-autonomic` adjusts {N, τ} → new `TopologyProvisionedEvent` |
| Topology selection (three frontiers) | Phase 2: roles[] → TeamSwarmHybrid; explicit kind field; auto from ParetoWeights + N vs N_max |
| Abstract AgentRole enum | `h2ai-types::AgentRole` — Coordinator / Executor / Evaluator / Synthesizer / Custom |
| Review gate (intra-swarm Evaluator gate) | Phase 3b: `ReviewGateTriggeredEvent` → Evaluator runs → approve or `ReviewGateBlockedEvent` |
| N_max^interface (Team-Swarm binding ceiling) | `crates/h2ai-autonomic` computes from CG(liaison, Coordinator); `InterfaceSaturationWarningEvent` + `h2ai_interface_n_max` metric |

### Task Manifest Parameters

The task manifest directly controls the three-axis Pareto position.

**Ensemble + CRDT (default):**
```json
{
  "description": "...",
  "pareto_weights": {
    "diversity": 0.5,     // weight on D axis
    "containment": 0.3,   // weight on E axis
    "throughput": 0.2     // weight on T axis
  },
  "topology": {
    "kind": "ensemble"    // explicit; or "auto" (default), "hierarchical_tree"
  },
  "explorers": {
    "count": 3,           // N — system caps at N_max if exceeded
    "tau_min": 0.1,       // lower bound on agent temperature
    "tau_max": 0.8        // upper bound — enforces τ diversity (Prop 3 condition 2)
  },
  "constraints": ["ADR-001", "ADR-004"],
  "context": "..."        // explicit dark knowledge (raises J_eff)
}
```

**Team-Swarm Hybrid (role-typed):**
```json
{
  "description": "...",
  "pareto_weights": {
    "diversity": 0.3,
    "containment": 0.4,
    "throughput": 0.3
  },
  "explorers": {
    "count": 4,
    "roles": [
      {"agent_id": "coord",     "role": "Coordinator"},
      {"agent_id": "worker_1",  "role": "Executor"},
      {"agent_id": "worker_2",  "role": "Executor"},
      {"agent_id": "checker",   "role": "Evaluator"}
    ],
    "review_gates": [
      {"reviewer": "checker", "blocks": "worker_1"},
      {"reviewer": "checker", "blocks": "worker_2"}
    ]
  },
  "constraints": ["ADR-001", "ADR-007"],
  "context": "..."
}
```

When `roles[]` is non-empty: topology is forced to `TeamSwarmHybrid`; `tau_min`/`tau_max` are ignored — each role's τ comes from the `AgentRole` canonical defaults (overridable per entry). The `review_gates[]` array declares which `Evaluator` Explorer must approve each `Executor` Explorer's output before it reaches the ADR Auditor.

The `tau_min` / `tau_max` spread (Ensemble mode) enforces Proposition 3's decorrelation condition: agents at τ = 0.1 and τ = 0.8 have substantially different sampling distributions and will hallucinate differently, so their errors are uncorrelated.

### Event Stream for a Successful Task (Ensemble + CRDT)

```
CalibrationCompletedEvent      → α=0.12, κ_eff=0.019, N_max=6.3
TaskBootstrappedEvent          → J_eff=0.71 (above 0.40 threshold), system_context compiled
TopologyProvisionedEvent       → topology_kind=Ensemble, N=3, merge_strategy=CrdtSemilattice
  (Phase 2.5: Multiplication Condition gate passes — no MultiplicationConditionFailedEvent)
ProposalEvent                  → explorer_id=exp_A, τ=0.15
ProposalEvent                  → explorer_id=exp_B, τ=0.50
ProposalEvent                  → explorer_id=exp_C, τ=0.80
GenerationPhaseCompletedEvent  → proposals_received=3, proposals_failed=0
  (Auditor validates each ProposalEvent as it arrives)
ValidationEvent                → explorer_id=exp_A
ValidationEvent                → explorer_id=exp_B
BranchPrunedEvent              → explorer_id=exp_C, violates=ADR-004 "budget from cache"
SemilatticeCompiledEvent       → merge_strategy=CrdtSemilattice, valid=2, pruned=1
MergeResolvedEvent             → resolution=select, selected=["exp_A","exp_B"]
```

---

## Worked Example: OAuth2 Authentication Service

A team of three engineers — a principal, a backend lead, and a security engineer — needs to deliver a new OAuth2 authentication service. Deliverables: implementation, security review, automated tests, and API documentation.

**Applying the four-question protocol:**

- **Q0:** Three engineers → Team-Swarm Hybrid
- **Q1:** Security service, auth tokens → Error cost = High
- **Q2:** Cryptographic primitive selection, token storage approach → Diversity = High
- **Q3:** Five specialized agents → N = 5 ≤ N_max = 6 → Ensemble viable within the swarm

**Swarm configuration** (domain context encoded in `system_context`; `AgentRole` is abstract):

| agent_id | AgentRole | τ | c_i | Function | Edge constraint |
|----------|-----------|---|-----|----------|-----------------|
| `coord` | `Coordinator` | 0.05 | 0.1 | Routes sub-tasks to leaf agents, summarises for liaison | Must be deterministic and auditable |
| `impl` | `Executor` | 0.4 | 0.5 | Implements auth logic, token handling, refresh flow | Gated by `evaluator` review gate |
| `evaluator` | `Evaluator` | 0.1 | 0.9 | OWASP check, cryptographic primitive review | **Review gate**: `{"reviewer":"evaluator","blocks":"impl"}` |
| `tester` | `Executor` (Custom τ=0.0) | 0.0 | 0.2 | Generates unit and integration tests | τ = 0 for error decorrelation from `impl` |
| `docs` | `Synthesizer` | 0.8 | 0.1 | API documentation, inline comments | High τ: diversity has value, errors easily corrected |

**Verifying the three N_max ceilings:**
- Human team: 3 engineers, N_max^human ≈ 10 ✓
- Agent swarm: 5 agents, N_max^swarm ≈ 6 ✓ (one agent of headroom)
- Interface: backend lead coordinates 3–4 concurrent sub-tasks, within liaison ceiling of ~5 ✓

**Dark knowledge to compile into the swarm coordinator's context:**
- bcrypt cost factor 12 (not the library default of 10)
- Session token storage in Redis, not in the database
- Security engineer's veto right on any cryptographic primitive choice
- The implicit definition of "production-ready" includes on-call coverage

**Outcome of role positioning (vs. "engineer with a chat window"):**

The `Evaluator` review gate converts a Byzantine fault (hallucinated OWASP compliance) from propagation-factor-of-4 to propagation-factor-of-1 — quarantined before it reaches the ADR Auditor. The tester's τ = 0 and different sampling distribution produces error decorrelation — it catches cases the `impl` Executor missed. The principal sees merged, pre-reviewed output, preserving human bandwidth for decisions that require consequence-awareness.

None of these effects require more agents. They require **positioned** agents.

---

## Summary

The topology decision reduces to two questions. First: does the task require a human team or a single human coordinator? If a team, the Team-Swarm Hybrid is the frontier topology. If a single human: is N above or below the ensemble capacity? Below N_max, use Ensemble with CRDT merge. Above N_max, extend to Hierarchical Tree.

Oracle, Star, Panel, and Pipeline are acceptable for simple or low-stakes tasks, but they all leave value on the table on at least one Pareto axis.

The practical ceiling that matters most is not the intra-swarm N_max or the human team N_max — it is the liaison's coordination ceiling: typically 3–5 concurrent swarm tasks. This is the binding constraint, and the answer to a saturated liaison is not to add more liaisons but to raise `CG(H_liaison, SC)` through better context compilation and ADR corpus quality.
