# USL Physics — The Mathematical Foundation

H2AI Control Plane is governed by the Universal Scalability Law, extended with an epistemic dimension that the original 1993 formulation did not need. This document defines every mathematical primitive the system uses and explains what each one means at the implementation level.

---

## The Universal Scalability Law

```
X(N) = N / (1 + α(N−1) + κ_eff · N(N−1))
```

`X(N)` is throughput at `N` agents. It has three regimes:

- **Linear growth** when both `α` and `κ_eff` are near zero — ideal, never real.
- **Amdahl plateau** when `α` dominates — throughput asymptotes to `1/α` regardless of N.
- **Throughput retrograde** when `κ_eff · N(N-1)` overtakes the linear term — adding agents makes things worse.

The system lives in the third regime risk zone whenever `N` approaches `N_max`. The MAPE-K loop exists to prevent crossing it.

---

## Contention Coefficient — α

**What it is:** The fraction of total work that is inherently serial. No parallelism can touch it.

**In this system:** The shared context window that only one agent can update at a time. The NATS subject lock during event publishing. The Merge Authority resolution step (one human, one decision).

**Measured by:** The calibration harness runs a set of representative tasks and measures the serial fraction empirically. Default reference for AI agent swarms: **α ≈ 0.10–0.15**.

**Implementation impact:** `h2ai_alpha` Prometheus gauge. Spikes indicate a shared resource bottleneck — context lock contention, token budget exhaustion, or a single slow Auditor blocking the pipeline.

---

## Coherency Coefficient — κ_base

**What it is:** The per-pair synchronization cost. Every time two nodes need to verify their private worlds agree, they pay κ. The total coherency cost grows as `κ · N(N-1)` — quadratic.

**In this system:** Token exchange overhead between agents. When Explorer A reads Explorer B's output to check for contradiction, that is κ. When the Auditor validates a proposal against the compiled Dark Knowledge, that is κ. When the state engine reconciles divergent proposals into a semilattice, that is κ.

**Measured by:** Calibration harness measures pairwise CG values across Explorer pairs on representative tasks. `κ_base` is the baseline before Common Ground adjustment.

**Default reference for AI agents:** **κ_base ≈ 0.01**.

**Why the event-sourced design zeroes α during generation:** Explorers append `ProposalEvent` and terminate. They do not read each other's output. `κ` during Phase 3 is structurally zero — the architecture enforces it.

---

## Common Ground Coefficient — CG(i, j)

```
CG(i, j) = J(K_i, K_j) × alignment(τ_i, τ_j)
```

Where:
- `J(K_i, K_j)` — Jaccard overlap of the knowledge sets of agents i and j.
- `alignment(τ_i, τ_j) = 1 − |τ_i − τ_j|` — how close their interpretive stances are.

**What it means:** Two agents with high CG share most of their knowledge and approach problems similarly. They are cheap to coordinate but bring little new information. Two agents with low CG disagree frequently — high coordination cost, but high epistemic diversity.

**The CG trade-off:** High CG reduces κ_eff (good for throughput) but also reduces the value of running multiple agents (bad for diversity). Low CG increases κ_eff but increases the information value of each proposal.

**Measured by:** The calibration harness runs representative tasks across Explorer pairs and measures agreement rates. CG samples are stored in `CoherencyCoefficients.cg_samples`.

---

## Effective Coherency — κ_eff

```
κ_eff = κ_base / mean(CG)
```

**What it means:** High Common Ground reduces coordination cost because agents with shared knowledge need fewer tokens to verify consistency. Low Common Ground amplifies κ_base because agents must do more work to bridge their private worlds.

**Implementation:** Computed by `CoherencyCoefficients::kappa_eff()` from calibration data. Exposed as `h2ai_kappa_eff` Prometheus gauge. The MAPE-K loop watches this gauge and triggers topology shifts when it approaches the threshold implied by `N_max`.

---

## Scalability Ceiling — N_max

```
N_max = sqrt((1 − α) / κ_eff)
```

**What it means:** The agent count at which `X(N)` peaks. Beyond `N_max`, throughput decreases with every additional agent — throughput retrograde.

**This is computable before spawning a single agent.** Given measured `{α, κ_base, CG}`, the system knows the ceiling before Phase 3 begins. This is the entire point of the calibration harness.

**Reference values:**

| Layer | α | κ_base | CG_mean | κ_eff | N_max |
|---|---|---|---|---|---|
| Hardware | 0.02 | 0.0003 | 1.0 | 0.0003 | ~57 |
| Human teams | 0.10 | 0.005 | 0.6 | 0.0083 | ~10 |
| AI agents | 0.15 | 0.01 | 0.4 | 0.025 | ~6 |

**Implementation:** `CoherencyCoefficients::n_max()`. Exposed as `h2ai_n_max` Prometheus gauge. If `N_requested > N_max`, the topology router automatically selects a Hierarchical Tree and reduces the effective N per sub-group.

---

## Coordination Threshold — θ_coord

```
θ_coord = min(CG_mean − σ_CG, 0.3)
```

**What it means:** The minimum CG value that any Explorer pair in the DAG must meet. If two agents are too epistemically distant (CG below θ_coord), their coordination cost is so high that they impede each other. The system either inserts a mediating Coordinator node or reduces N.

**Computed from:** Calibration data — `CoordinationThreshold::from_calibration(&cc)`. The formula uses one standard deviation below the mean as the floor, capped at 0.3.

**Implementation:** Stored in `TopologyProvisionedEvent.coordination_threshold`. Exposed as `h2ai_theta_coord` Prometheus gauge. When `CG_mean` approaches `θ_coord`, the operator sees the swarm is near the coordination floor and should either narrow τ spread or add a Coordinator.

---

## Dark Knowledge Gap — J_eff

```
J_eff = J(K_prompt, K_task_required)
```

**What it means:** The Jaccard overlap between what the human explicitly provided in the manifest (`K_prompt`) and what the task actually requires (`K_task_required`). When J_eff is low, agents must guess tacit constraints — they hallucinate architectural decisions the human never externalized.

**The Dark Knowledge problem:** In a team environment, humans hold vast tacit knowledge about constraints: "we use stateless auth because of compliance requirement X," "this service cannot touch the database directly," "ADR-007 forbids this pattern." When these constraints are not compiled into `system_context`, they appear as Byzantine faults from the Auditor's perspective — valid-looking outputs that violate invisible rules.

**Implementation:** Computed by `crates/context` from the submitted manifest and the ADR corpus. If `J_eff < threshold` → synchronous `ContextUnderflowError` returned to the API caller. Nothing is written to NATS. `J_eff` is recorded in `TaskBootstrappedEvent` and exposed as `h2ai_j_eff` Prometheus gauge.

---

## Role Error Cost — c_i

**What it is:** A weight `c_i ∈ [0, 1]` representing the damage if role `i` produces a Byzantine error that reaches the human.

**Examples:**
- `c_i ≈ 0.1` — Early Explorer draft (low stakes, Auditor will filter it)
- `c_i ≈ 0.7` — Swarm Coordinator output (medium stakes, error multiplied across sub-group)
- `c_i ≈ 0.9` — Auditor false positive (catastrophic — a hallucination passed as valid to the Merge Authority)

**Implementation:** Carried in `TopologyProvisionedEvent.role_error_costs`. When a proposal is pruned, `BranchPrunedEvent.constraint_error_cost` records the c_i of the violated constraint, so the Merge Authority UI can show the stakes of each rejection.

---

## Merge Strategy Selection

```
MergeStrategy = CrdtSemilattice  if max(c_i) ≤ 0.85
              = BftConsensus      if max(c_i) > 0.85
```

**CRDT Semilattice (AP):** Mathematically joins divergent proposals without synchronous coordination. Preserves epistemic diversity — all surviving proposals contribute to the final diff. `O(1)` reconciliation cost. Used when the cost of a Byzantine error is tolerable (caught and corrected by the human in the Merge Authority).

**BFT Consensus (CP):** Byzantine-fault-tolerant consensus over surviving proposals before human presentation. Higher κ cost, but provides mathematical safety when a single undetected error would be catastrophic. Used when `max(c_i) > 0.85` — the Auditor role's error cost exceeds the safety threshold.

---

## Multiplication Condition (Proposition 3)

For collective performance to exceed individual performance, **all three conditions must hold simultaneously:**

**Condition 1 — Baseline competence:**
```
p_correct > 0.5  for every planned Explorer on the calibration task set
```
If any Explorer is performing worse than random chance on the representative tasks, adding it makes the collective worse. Measured in Phase 0.

**Condition 2 — Error decorrelation:**
```
ρ(err_i, err_j) < 0.9  for every Explorer pair
```
If two Explorers make the same errors 90%+ of the time, they provide no independent signal. They are structurally redundant. Fix: widen τ spread or route some Explorers to different model backends. Measured in Phase 0.

**Condition 3 — Common Ground floor:**
```
CG_mean ≥ θ_coord
```
Agents must share enough ground to coordinate without catastrophic overhead. If they are too epistemically distant, the coordination cost exceeds the diversity benefit.

**This is a hard gate, not a gradient.** `MultiplicationConditionFailedEvent` re-enters Phase 2. The autonomic loop adjusts parameters and retries. The gate is enforced before a single inference token is generated.
