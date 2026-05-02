# Agent Descriptor Guide

## What an Agent Is

In H2AI, an **edge agent** is any LLM-based stateless container that:

1. Receives a `TaskPayload` over NATS JetStream
2. Runs inference (with or without tool calls) and produces a `TaskResult`
3. Publishes telemetry to `h2ai.telemetry.{task_id}` while running
4. Terminates — it has no persistent state

An agent is **not** a named product, not a model family, and not a version. It is described entirely by two attributes:

```rust
pub struct AgentDescriptor {
    pub model: String,         // "llama3-70b", "gpt-4o", "claude-3-opus", ...
    pub tools: Vec<AgentTool>,
}

pub enum AgentTool {
    Shell,
    WebSearch,
    CodeExecution,
    FileSystem,
}
```

`model` selects the inference backend. `tools` declares what capabilities the container has been granted for this task. These are independent: any model can run with any tool set.

---

## Why Generalization Matters

The previous design used `AgentType { OpenClaw, NeoClaw }` — named variants for specific container products. This caused three problems:

**1. The orchestrator had to know which image to pull.** Every new agent type required a code change to add a variant and a mapping from that variant to a container image. With `AgentDescriptor`, the provider reads `descriptor.model` and constructs the image tag from configuration, not from an enum.

**2. The math could not be applied correctly.** The system computes `c_i` (role error cost), `α` (serial contention fraction), and `κ_base` (coherency cost) per agent — and these quantities depend on what tools the agent has. A named type hides this; a descriptor exposes it. The topology planner, merge strategy selector, and MAPE-K loop all read from the descriptor to set physics parameters.

**3. Tool capability was implied, not declared.** "NeoClaw" agents happened to have shell access. This was tribal knowledge. With `AgentTool`, the container's security context, volume mounts, and NATS NKey permissions are derived automatically from the declared tools — no manual cross-referencing.

---

## Pure LLM vs. Tool-Using Agent — Mathematical Properties

### Pure LLM Agent (`tools: []`)

A pure LLM agent is a deterministic function:

```
output = f(system_context, instructions, τ)
```

The only degree of freedom is temperature `τ`. This has direct consequences for the USL extension:

**α ≈ 0 during generation.** The agent acquires no shared locks. It reads context once, generates in isolation, and terminates. N pure LLM agents running in parallel contribute zero to α during the generation phase.

**Error independence holds.** Two agents with different τ values produce outputs from independent distributions. This satisfies the error decorrelation requirement in Proposition 3 (Multiplication Condition): `ρ_errors < max_error_correlation`.

**c_i is low (0.1–0.3).** A wrong text output costs nothing to discard. The Auditor gate tombstones it with zero collateral damage. CRDT semilattice merge applies.

**CG spread is controllable via τ.** Common Ground between agents is a function of τ difference. By spreading τ from `tau_min` to `tau_max`, the calibration harness can confirm that CG stays below θ_coord — ensuring diversity without explicit role declarations.

### Tool-Using Agent (`tools: [Shell, ...]`)

A tool-using agent is:

```
output = f(system_context, instructions, τ, external_state_at_time_T)
               + side_effects
```

The `external_state_at_time_T` dependency breaks assumptions:

**α increases.** Tool calls introduce serialization. A `Shell` command running `git commit` locks the repository. A `FileSystem` write to a shared path serializes concurrent agents. The calibration harness measures the resulting α — a pool with tool-using agents will have α in the 0.20–0.30 range rather than 0.10–0.15, which lowers `N_max` and causes the planner to provision fewer explorers. This is correct behavior: the system adjusts scope to match the actual parallelism available.

**κ_base increases from retrieval nondeterminism.** Two agents calling `WebSearch("current API rate limits")` at different moments may receive different results. Their outputs diverge for environmental reasons, not reasoning differences. This inflates CG variance, which raises `κ_eff = κ_base / CG_mean`, which again lowers `N_max`.

**Error independence weakens.** If two agents both call a flaky external API and both fail, their errors are correlated. The Multiplication Condition checks `ρ_errors < max_error_correlation` (default 0.9). Tool-induced correlated failures can push ρ toward this threshold and trigger `MultiplicationConditionFailedEvent`, causing the MAPE-K loop to retry with a reduced explorer count or adjusted τ spread.

**c_i is high.** Wrong tool-using outputs may have irreversible side effects. This drives `max(c_i) > 0.85`, switching `MergeStrategy` from CRDT to BFT. It also makes Review Gates mandatory in TeamSwarmHybrid topology: the Evaluator (pure LLM, τ≈0.1) receives each Executor proposal before it reaches the ADR Auditor. A blocked proposal emits `ReviewGateBlockedEvent` and is tombstoned — the bad output cannot reach the human by graph construction.

---

## Tool Set to Physics Parameter Mapping

| Tool set | Effect on α | Effect on κ_base | Default c_i | Suggested role | τ range |
|---|---|---|---|---|---|
| `[]` | +0 | +0 | 0.1–0.3 | Coordinator, Synthesizer | 0.05–0.9 |
| `[WebSearch]` | +0.01–0.02 | +0.005 | 0.2–0.4 | Evaluator | 0.05–0.3 |
| `[FileSystem]` | +0.02–0.05 | +0.010 | 0.4–0.6 | Executor | 0.3–0.6 |
| `[CodeExecution]` | +0.03–0.08 | +0.015 | 0.5–0.7 | Executor | 0.3–0.6 |
| `[Shell]` | +0.05–0.15 | +0.020 | 0.6–0.9 | Executor | 0.3–0.5 |
| `[Shell, CodeExecution, FileSystem]` | +0.08–0.20 | +0.025 | 0.7–0.9 | Executor | 0.3–0.5 |

These are default ranges. The calibration harness measures actual α and CG for your specific adapter pool. The table is a prior — the harness is ground truth.

**BFT trigger:** If any agent in the topology has `c_i > 0.85`, or if the Executor role's assigned c_i from the manifest pushes `max(c_i) > 0.85`, `MergeStrategy` switches to BFT. You can observe this as `ConsensusRequiredEvent` in the task stream.

---

## Examples

### Example 1 — Architecture review (pure LLM)

Three reasoning agents explore different τ values to maximize proposal diversity. No tools — α stays near reference range, CRDT merge applies.

```json
{
  "description": "Propose a stateless JWT auth rotation strategy consistent with ADR-001",
  "pareto_weights": {"diversity": 0.6, "containment": 0.3, "throughput": 0.1},
  "explorers": {
    "count": 3,
    "tau_min": 0.2,
    "tau_max": 0.85
  },
  "constraints": ["ADR-001"]
}
```

What happens:
- Three pure LLM Executor agents at τ = 0.2, 0.525, 0.85
- α stays in 0.10–0.15 reference range → N_max ≈ 5–7 → 3 explorers well within ceiling
- c_i ≈ 0.1–0.3 for all → `MergeStrategy::CrdtSemilattice`
- No Review Gates needed — Auditor receives proposals directly
- Human resolves one CRDT diff via Merge Authority

### Example 2 — Code generation with tool-using executors

Executor agents run and test code. An Evaluator (pure LLM) forms a Review Gate before the Auditor, because c_i for CodeExecution agents may push toward the BFT threshold.

```json
{
  "description": "Write and test a Redis Lua script for atomic budget check-and-decrement with 30s TTL idempotency",
  "pareto_weights": {"diversity": 0.2, "containment": 0.7, "throughput": 0.1},
  "explorers": {
    "roles": [
      {"agent_id": "executor_A", "role": "Executor", "tau": 0.4},
      {"agent_id": "executor_B", "role": "Executor", "tau": 0.5},
      {"agent_id": "evaluator",  "role": "Evaluator", "tau": 0.1}
    ],
    "review_gates": [
      {"reviewer": "evaluator", "blocks": "executor_A"},
      {"reviewer": "evaluator", "blocks": "executor_B"}
    ]
  },
  "constraints": ["ADR-004", "ADR-007"]
}
```

What happens:
- Executor agents receive `AgentDescriptor { model: "...", tools: [CodeExecution, FileSystem] }` — their containers get a workspace mount and code execution sandbox
- Evaluator receives `AgentDescriptor { model: "...", tools: [] }` — pure LLM, τ=0.1, reads Executor output and checks it against constraints
- Executor c_i ≈ 0.65–0.70 per the tool set → `MergeStrategy::BftConsensus` if max(c_i) > 0.85; otherwise CRDT
- α measured by calibration will be higher (0.18–0.25) due to code execution serialization → N_max is lower → system confirms 2 Executors + 1 Evaluator fits within ceiling
- `ReviewGateBlockedEvent` tombstones Executor proposals the Evaluator rejects before they reach the ADR Auditor

### Example 3 — Research synthesis (WebSearch + pure LLM)

A Coordinator reasons about what to search; Search agents retrieve; a Synthesizer (pure LLM, high τ) combines.

```json
{
  "description": "Research current best practices for distributed rate limiting and summarize trade-offs",
  "pareto_weights": {"diversity": 0.4, "containment": 0.4, "throughput": 0.2},
  "explorers": {
    "roles": [
      {"agent_id": "searcher_A", "role": "Executor", "tau": 0.3},
      {"agent_id": "searcher_B", "role": "Executor", "tau": 0.3},
      {"agent_id": "synthesizer", "role": "Synthesizer", "tau": 0.8}
    ]
  }
}
```

What happens:
- Searcher agents receive `AgentDescriptor { tools: [WebSearch] }` — NKey permits outbound web calls
- Synthesizer receives `AgentDescriptor { tools: [] }` — pure LLM, high τ for creative synthesis
- WebSearch c_i ≈ 0.3 → `MergeStrategy::CrdtSemilattice` — nondeterminism is managed at the Auditor gate, not BFT
- κ_eff slightly elevated due to retrieval variance → system provisions conservatively (fewer parallel searchers than a pure-LLM task of the same size)

---

## How the Framework Uses the Descriptor

**`TopologyPlanner::provision` (autonomic crate)**

Reads `descriptor.tools` per role and applies default c_i if none is declared in the manifest. Calls `MergeStrategy::from_role_costs` with all c_i values. If any explorer is assigned `Shell` or `CodeExecution`, the planner checks whether a Review Gate exists in the topology — if TeamSwarmHybrid is selected and no gate is declared, it logs a warning (future: hard gate).

**`KubernetesProvider::ensure_agent_capacity` (h2ai-provisioner crate)**

Maps `descriptor.model` → container image (e.g. `registry/agent:llama3-70b`). Maps `descriptor.tools` → volume mounts and security contexts:
- `Shell` → writable workspace mount, `capabilities: {add: ["SYS_PTRACE"]}` in the Pod spec
- `CodeExecution` → isolated sandbox volume, resource limits on CPU/memory
- `FileSystem` → writable shared workspace mount
- `WebSearch` → egress NetworkPolicy allowing outbound HTTPS
- `[]` → no additional mounts, no additional capabilities — minimal attack surface

**`h2ai-nats::nkey::generate_task_nkey` (h2ai-nats crate)**

Scopes the NKey `allowed_publish` list to match the tool set. A pure LLM agent gets three subjects: `h2ai.telemetry.{agent_id}`, `audit.events.{agent_id}`, `h2ai.results.{task_id}`. A Shell agent additionally gets `h2ai.telemetry.{agent_id}.shell.*` for structured shell telemetry. No agent can publish to subjects it was not explicitly granted — NATS server enforces this.

**`CalibrationHarness::run` (autonomic crate)**

Runs calibration tasks against the actual adapter pool. If the pool contains tool-using adapters, the measured α reflects their serialization costs. The `N_max` computed from calibration is automatically adjusted — no manual correction needed. The harness reports `CoherencyCoefficients { alpha, kappa_base, cg_samples }`, and `N_max = sqrt((1-alpha) / kappa_eff)` already accounts for tool overhead.

---

## Frequently Asked Questions

**Q: How does the system know which container image to use?**

The `KubernetesProvider` (or `StaticProvider`) reads `AgentDescriptor.model` and looks up the image in the provider's configuration (environment variable or configmap). The orchestrator never hard-codes image names — it passes the descriptor to the provider, and the provider's configuration maps model names to images.

**Q: Can I use different models for different explorers in the same task?**

Yes. Each explorer in `explorers.roles[]` carries an `AgentDescriptor` independently. You can have `executor_A` running `llama3-70b` and `executor_B` running `gpt-4o` in the same task. The calibration harness should include both adapters in the pool when measuring α and CG.

**Q: When does max(c_i) > 0.85 actually get reached?**

Typically when you have Executor agents with `Shell` + `CodeExecution` + `FileSystem` (c_i ≈ 0.85–0.90) or when the manifest explicitly assigns `role_error_cost: 0.9` to a high-risk Evaluator. BFT consensus is not common in practice — it is a safety net for tasks where conflicting outputs represent genuinely irreconcilable divergent states.

**Q: Why is the Evaluator's c_i also high (0.9) when it has no tools?**

The Evaluator's c_i is not about tool destructiveness — it is about the cost of a wrong gating decision. An Evaluator with c_i=0.9 that incorrectly blocks a valid Executor proposal wastes significant work. An Evaluator that incorrectly approves a bad Executor proposal allows a potentially dangerous output to reach the Auditor. The high c_i reflects gating authority, not tool risk.

**Q: Does the model name affect any math?**

No — the model name is opaque to the physics layer. Only `tools` affects α, κ_base, and c_i directly. Model choice affects the quality of outputs and therefore the empirical CG samples measured during calibration, but that effect is captured by the harness, not hard-coded.
