# Mathematics Apparatus

This document is the formal reference for every quantitative claim made by the H2AI Control Plane. All runtime decisions — spawning agents, selecting merge semantics, gating tasks, retrying topologies — are implementations of the theorems here.

The theory is an extension of Gunther's Universal Scalability Law (USL) to systems where nodes have private epistemic states. The classical USL treats all nodes as uniform. The extension relaxes this assumption: nodes carry different knowledge bases and different temperature parameters, and the coherency cost between any pair of nodes depends on how much of their knowledge overlaps.

### Validation scripts

Every definition and proposition in this document has a corresponding numerical check or simulation. **If you change a formula, constant, or threshold here, you must update both scripts and re-run them before merging.**

| Script | Purpose | Run |
|--------|---------|-----|
| [`scripts/validate_math.py`](../../scripts/validate_math.py) | Asserts every definition and proposition numerically; stdlib only; CI-runnable | `python scripts/validate_math.py` |
| [`scripts/simulate_usl.py`](../../scripts/simulate_usl.py) | Plots USL curves, CG_mean sensitivity, Pareto matrix, J_eff gate; requires numpy + matplotlib | `python scripts/simulate_usl.py` |

> **Sync rule:** The calibration constants (`α`, `κ_base`, `CG_mean`, `N_max`) in `CALIBRATION_TABLE` (validate) and `LAYERS` (simulate) must exactly match §3 of this document. The J_eff gate value (`J_EFF_GATE = 0.4`) must match §4. The BFT threshold (`BFT_THRESHOLD = 0.85`) must match Proposition 5.

---

## 1. Definitions

### Definition 1 — Contention Coefficient α

The fraction of a workload that is irreducibly serial. Formally, if `S` is the set of operations that cannot proceed in parallel:

```
α = |S| / |W|   where W is the complete workload
```

**Bounds:** `0 ≤ α < 1`. At `α = 0` the workload is embarrassingly parallel. At `α = 1` parallelism provides no benefit (pure Amdahl).

**Calibrated values:**

| Layer       | Typical α  |
|-------------|-----------|
| CPU cores   | 0.02      |
| Human teams | 0.10      |
| AI agents   | 0.15      |

In H2AI, α is measured during calibration as the fraction of task steps that require the orchestrator's exclusive attention (merge authority decisions, auditor gates, MAPE-K retries).

> **Script cross-reference:** α is a parameter in every USL call — see `validate_math.py::usl_throughput()` and `simulate_usl.py::usl()`. Layer values are stored in `CALIBRATION_TABLE` / `LAYERS` — see §3.

---

### Definition 2 — Coherency Coefficient κ

The per-pair synchronisation cost: the fraction of capacity spent on one pair of nodes reaching a mutually consistent state. The classical USL throughput formula is:

```
X(N) = N / (1 + α(N − 1) + κ · N(N − 1))
```

where `N` is the number of agents. The `κ · N(N − 1)` term is quadratic — it is the origin of the scalability wall.

**Calibrated values:**

| Layer       | Typical κ_base |
|-------------|---------------|
| CPU cores   | 0.0003        |
| Human teams | 0.005         |
| AI agents   | 0.01          |

> **Script cross-reference:**
> - `validate_math.py` — `usl_throughput(N, alpha, kappa)`: checks `X(1) = 1` for all layers; checks `X(N) > 0` for N∈[1,20]
> - `simulate_usl.py` — `usl(N, alpha, kappa)`: Plot 1 (three-layer USL curves)

---

### Definition 3 — Common Ground CG(i, j)

A measure of the epistemic overlap between agents `i` and `j`. It has two components:

```
CG(i, j) = J(K_i, K_j) × alignment(τ_i, τ_j)
```

where:
- `J(K_i, K_j) = |K_i ∩ K_j| / |K_i ∪ K_j|` is the Jaccard similarity of the agents' knowledge bases
- `alignment(τ_i, τ_j) ∈ (0, 1]` is a monotonically decreasing function of `|τ_i − τ_j|` — agents with very different temperatures share less interpretive frame even when they share vocabulary

**Bounds:** `0 < CG(i, j) ≤ 1`. Equal to 1 only when the agents are identical in both knowledge and temperature (zero diversity, no coordination benefit).

**Mean common ground:**

```
CG_mean = (2 / N(N−1)) · Σ_{i<j} CG(i, j)
```

> **Script cross-reference:**
> - `validate_math.py` — `jaccard(set_a, set_b)`, `tau_alignment(tau_i, tau_j)`, `common_ground(K_i, K_j, tau_i, tau_j)`: checks identity (CG=1), disjoint sets (CG=0), symmetry, temperature divergence effect
> - `simulate_usl.py` — `kappa_eff(kappa_base, cg_mean)` uses CG_mean; Plot 2 (CG_mean sensitivity on N_max)

---

### Definition 4 — Effective Coherency κ_eff

The actual coherency cost paid by a system, accounting for the epistemic structure of its agents:

```
κ_eff = κ_base / CG_mean
```

When agents share more common ground, each synchronisation step costs less wall-clock capacity — the coordination messages carry more mutual information per token. When common ground is low (agents have divergent knowledge or temperatures), the same coherency requirement demands more messages.

**Key implication:** κ_eff can vary by an order of magnitude across layers. CPU cores at `CG = 1.0` pay `κ_eff = κ_base`. AI agents at `CG_mean = 0.4` pay `κ_eff = κ_base / 0.4 = 2.5 × κ_base`. The wall arrives 2.5× earlier.

> **Script cross-reference:**
> - `validate_math.py` — section "Definition 4": asserts `κ_eff = κ_base / CG_mean` for all three calibration layers; asserts higher CG → lower κ_eff invariant
> - `simulate_usl.py` — `kappa_eff(kappa_base, cg_mean)`: used in every plot; Plot 2 left panel shows κ_eff → N_max relationship explicitly

---

### Definition 5 — Extended USL

The throughput function extended to account for epistemic structure:

```
X(N, τ, K) = N / (1 + α(N − 1) + κ_eff · N(N − 1))
```

where `κ_eff = κ_base / CG_mean(τ, K)` is itself a function of the agent temperature vector `τ` and knowledge matrix `K`. This is the formula H2AI optimises at calibration time.

> **Script cross-reference:**
> - `validate_math.py` — `usl_throughput_extended(N, alpha, kappa_base, cg_mean)`: verifies algebraic equivalence of the two N_max forms (§ Proposition 1 calibration cross-check)
> - `simulate_usl.py` — `usl(N, alpha, kappa)` with `kappa = kappa_eff(kb, cg)`: Plot 2 right panel shows three CG_mean values producing different USL curves from the same κ_base

---

### Definition 6 — Coordination Edge Count

The number of pairwise coordination edges in a topology:

```
E_flat = N(N − 1) / 2        (fully connected mesh)
E_tree = N − 1               (tree, one edge per non-root node)
```

The ratio `E_flat / E_tree = (N − 1) / 2` grows linearly with `N`. At `N = 6`, a flat mesh has 15 edges; a tree has 5. This is why the tree topology scores significantly higher on throughput at large `N` — it converts quadratic overhead to linear overhead.

> **Script cross-reference:**
> - `validate_math.py` — `edge_count_flat(n)`, `edge_count_tree(n)`: asserts `E_tree < E_flat` for N ∈ {3, 5, 6, 10}; used in Proposition 5 proof verification

---

### Definition 7 — Role-Weighted Interaction Graph

A weighted directed graph `G = (V, E, w)` where:
- Vertices `V` are agents
- Edges `E` are coordination dependencies
- Edge weight `w(i → j) = freq(i, j) × c_i × c_j × κ_eff`

where `c_i ∈ [0, 1]` is the **role error cost** of agent `i` — the expected cost if agent `i` produces a Byzantine output that propagates to the merge authority unchallenged.

High `c_i` agents (security reviewer, financial auditor) should have outbound edges that terminate at a quarantine gate, not directly at the human merge authority.

> **Script cross-reference:**
> - `validate_math.py` — `c_i` is used as a parameter in `byzantine_loss()` (Definition 8) and `merge_strategy()` (Proposition 5). The graph structure is implicitly validated through the propagation factor in Definition 8 checks.
> - `simulate_usl.py` — `TOPOLOGIES` list encodes the Pareto scores that result from different graph structures (Plot 3).

---

### Definition 8 — Byzantine Expected Loss

The expected loss from a Byzantine fault originating at agent `i`:

```
L_i = c_i × P(hallucination_i) × propagation(topology)
```

where `propagation(topology)` is:
- `N − 1` for a flat topology (no gate; the fault reaches every peer)
- `k` (branching factor) for a tree (fault reaches only `k` children before the coordinator intercepts)
- `1` for a quarantine gate (the gate blocks the fault before it exits the subtree)

The total expected loss of a topology is `Σ_i L_i`. Topology selection minimises this sum subject to throughput and diversity constraints.

> **Script cross-reference:**
> - `validate_math.py` — `byzantine_loss(c_i, p_hallucination, propagation)`: asserts `L(flat) > L(tree) > L(gate)` for N=5; asserts total fleet loss flat > tree
> - `simulate_usl.py` — containment scores (E column) in `TOPOLOGIES` are derived from this formula; visible in Plot 3

---

### Definition 9 — Three-Axis Pareto Frontier

Every topology is evaluated on three objective functions simultaneously:

| Axis            | Symbol | Definition |
|-----------------|--------|------------|
| **Throughput**  | T      | `X(N) / X(1)` — normalised throughput relative to single agent |
| **Containment** | E      | `1 − (E[L] / E[L_max])` — how well the topology limits error propagation |
| **Diversity**   | D      | `H(τ) / H_max` — normalised entropy of the agent temperature distribution |

A topology is **Pareto non-dominated** if no alternative topology beats it on all three axes simultaneously. The H2AI Pareto Frontier contains three topologies: Ensemble + CRDT (single human, small N), Hierarchical Tree (single human, large N), and Team-Swarm Hybrid (team scale).

> **Script cross-reference:**
> - `validate_math.py` — `entropy(probs)` implements `H(τ)`; used in Proposition 4 checks
> - `simulate_usl.py` — `TOPOLOGIES` constant contains the (T, E, D, frontier) tuples; Plot 3 renders the full matrix. **If you change any score in `TOPOLOGIES`, update the corresponding row in `docs/guides/theory-to-implementation.md` Pareto Summary table.**

---

### Definition 10 — Dark Knowledge Gap

The Jaccard distance between the knowledge explicitly available in a task submission and the knowledge actually required to produce a sound architectural proposal:

```
J_eff(K_prompt, K_task_required) = |K_prompt ∩ K_task_required| / |K_prompt ∪ K_task_required|
```

`J_eff ∈ [0, 1]`. The H2AI gate threshold is `J_eff_min = 0.4`. Submissions with `J_eff < 0.4` return `ContextUnderflowError` — the constraint space is too underspecified for the Auditor to enforce meaningfully.

ADRs are the primary mechanism for increasing `J_eff`. Each ADR's `## Constraints` section contributes explicit prohibition and requirement statements to `K_prompt`.

> **Script cross-reference:**
> - `validate_math.py` — `j_eff(k_prompt, k_required)` reuses `jaccard()`; `J_EFF_GATE = 0.4` must equal this threshold; checks: full coverage=1.0, empty prompt < gate, 2-of-4 coverage ≥ gate, monotonic growth with ADR count
> - `simulate_usl.py` — `J_EFF_GATE = 0.4` must equal this threshold; Plot 4 (J_eff distribution and acceptance rate vs ADR corpus size)

---

## 2. Propositions

### Proposition 1 — Scalability Ceiling

**Statement:** The throughput function `X(N)` has a unique maximum at:

```
N_max = sqrt((1 − α) / κ_eff)
```

For the extended USL with epistemic structure:

```
N_max = sqrt((1 − α) · CG_mean / κ_base)
```

**Proof sketch:** Differentiate `X(N)` with respect to `N` and set to zero:

```
dX/dN = [1 + α(N−1) + κ·N(N−1) − N(α + κ(2N−1))]
         / [1 + α(N−1) + κ·N(N−1)]²  = 0

Numerator = 0 ⟹ 1 − α − κ(N² − 1/... [algebra]) = 0
                ⟹ N² = (1 − α) / κ
                ⟹ N_max = sqrt((1 − α) / κ)
```

Adding agents beyond `N_max` causes retrograde — throughput decreases as the quadratic coherency term `κ·N(N−1)` dominates the linear productivity term `N`.

**Calibrated ceilings:**

| Layer       | α    | κ_base | CG_mean | κ_eff  | N_max |
|-------------|------|--------|---------|--------|-------|
| CPU cores   | 0.02 | 0.0003 | 1.0     | 0.0003 | ≈ 57  |
| Human teams | 0.10 | 0.005  | 0.6     | 0.0083 | ≈ 10  |
| AI agents   | 0.15 | 0.01   | 0.4     | 0.025  | ≈ 6   |

> **Script cross-reference:**
> - `validate_math.py` — `analytical_n_max(alpha, kappa)` implements `sqrt((1-α)/κ)`; `numerical_n_max(alpha, kappa)` finds the peak by exhaustive search and asserts it matches within ±2; sign-change test confirms dX/dN passes through zero; retrograde asserted as `X(2·N_max) < X(N_max)`; algebraic equivalence of both N_max forms asserted in calibration cross-check
> - `simulate_usl.py` — `n_max(alpha, kappa)`: N_max verticals on Plot 1; left panel of Plot 2 shows N_max as a continuous function of CG_mean

---

### Proposition 2 — Epistemic Conway Constraint

**Statement:** A team-to-system homomorphism is valid (the system architecture mirrors the team structure without information loss) if and only if every pair of adjacent team members `(i, j)` satisfies:

```
CG(i, j) ≥ θ_coord
```

where the coordination threshold is:

```
θ_coord = min(CG_mean − σ_CG, 0.3)
```

(`σ_CG` is the standard deviation of the pairwise CG distribution across the team.)

**Implication:** When two humans who need to coordinate on a system boundary have `CG(i, j) < θ_coord`, the boundary is misplaced — they share insufficient knowledge to produce a coherent interface. The fix is either to raise `CG` (shared ADRs, knowledge transfer) or to shift the boundary to a pair with higher common ground.

In H2AI, `θ_coord` is computed from the calibration measurements and stored as `coordination_threshold` in the system state. An Explorer spawn is rejected if the spawned agent's `CG` with the existing topology falls below this threshold.

> **Script cross-reference:**
> - `validate_math.py` — `coordination_threshold(cg_values)`: implements `min(mean - σ, 0.3)`; checks: well-aligned team passes (all CG ≥ θ), misaligned pair is detected (min CG < θ)

---

### Proposition 3 — Multiplication Condition (Generalised Condorcet)

**Statement:** A multi-agent system produces outcomes strictly better than its best single member if and only if all three conditions hold simultaneously:

1. **Competence:** Every agent `i` has individual accuracy `p_i > 0.5` on the task class.
2. **Decorrelation:** No pair of agents `(i, j)` has error correlation `ρ_{ij} ≥ 0.9`. (Correlated agents make the same mistakes; adding them adds cost without adding diversity.)
3. **Coordination viability:** `CG_mean ≥ θ_coord`. (Agents must share enough common ground to produce a coherent merged output.)

**LLM violation note:** Condition 1 is frequently violated by LLMs on tasks outside their training distribution. Condition 2 is violated when all agents use the same base model at similar temperatures — they share hallucination modes. H2AI enforces both through calibration: `p_i` is estimated per task class, and `τ` diversity is enforced by the Explorer provisioning protocol (`tau_min` / `tau_max` task parameters).

**Consequence for merge semantics:** When any condition fails, adding more agents does not produce the Condorcet multiplication effect. The system should reduce `N` rather than increase it.

> **Script cross-reference:**
> - `validate_math.py` — `majority_vote_accuracy(p, n_agents)`: exact binomial sum; asserts p=0.7 ensemble beats individual, p=0.4 ensemble is worse (Condition 1); asserts monotonic accuracy growth with N at p=0.7; `correlated_ensemble_accuracy(p, n, rho)`: asserts ρ=0.95 reduces ensemble benefit (Condition 2)

---

### Proposition 4 — Merge Semantics and Epistemic Entropy

**Statement:** Let `H(τ)` be the Shannon entropy of the agent temperature distribution before merge:

```
H(τ) = −Σ_i p(τ_i) · log p(τ_i)
```

- **Consensus merge** collapses `H(τ) → 0`. All agents align to the majority view. The merged output has zero remaining epistemic diversity — minority positions are discarded.
- **CRDT merge** preserves `H(τ)` to the merge authority. All contributions survive to the human decision point. Entropy is resolved by human judgement, not by a voting rule.

**Implication:** Consensus is appropriate when the task has a single verifiable correct answer (deterministic transformation, lookup). CRDT merge is appropriate when the task value derives from diverse strategies reaching the merge authority intact (architecture, security review, root-cause analysis).

Forcing consensus on a high-diversity task discards the information that makes multi-agent systems valuable.

> **Script cross-reference:**
> - `validate_math.py` — `entropy(probs)`: Shannon entropy; asserts consensus → H≈0, CRDT → H preserved, H(CRDT) > H(consensus)

---

### Proposition 5 — CRDT-Merge Hierarchy Dominance

**Statement:** For any fixed `N`, a hierarchical topology with CRDT merge semantics at each internal node is Pareto non-dominated on `(T, E, D)` relative to any flat topology.

**Proof sketch:**
- **T (Throughput):** The tree converts `O(N²)` coordination edges to `O(N)` edges, strictly increasing throughput for `N > N_max(flat)`.
- **E (Containment):** Each internal node acts as a quarantine gate. A Byzantine fault propagates to at most `k` peers (branching factor) rather than `N − 1` peers in a flat mesh.
- **D (Diversity):** CRDT merge preserves all leaf contributions to each internal node, and the internal node forwards all contributions (not a consensus-collapsed single view) to the next level.

**Safety constraint:** The CRDT-merge guarantee applies only when no individual agent `i` is a Byzantine actor. When role error cost `c_i > 0.85`, CRDT merge is insufficient — a single high-error-cost agent can poison the semilattice. The safety constraint is:

```
if max(c_i) ≤ 0.85 → MergeStrategy = CrdtSemilattice
if max(c_i) >  0.85 → MergeStrategy = BftConsensus
```

H2AI evaluates this condition at the point of `SemilatticeCompiledEvent` generation. If the condition flips to BFT mid-task (e.g., a proposal's role error cost is recalculated after audit), the merge is re-run under BFT semantics.

> **Script cross-reference:**
> - `validate_math.py` — `edge_count_flat/tree()`: asserts E_tree < E_flat for N∈{3,5,6,10}; tree N_max > flat N_max (same α, reduced κ_eff); `merge_strategy(role_error_costs)`: asserts CrdtSemilattice at max(c_i)=0.85, BftConsensus at max(c_i)=0.851 — **boundary is exact, do not change 0.85 without updating `BFT_THRESHOLD` in validate_math.py**
> - `simulate_usl.py` — `TOPOLOGIES` frontier flags encode the Pareto non-dominance claim; Plot 3 makes it visually verifiable

---

## 3. Calibration

Calibration establishes the system-specific values of `α`, `κ_base`, and `CG_mean` by running a structured set of test tasks and measuring actual throughput at varying `N`. The three-curves method fits the USL formula to three throughput measurements at `N = 1`, `N = 3`, and `N = 6`:

```
α = (3·X(3) − X(1) − 3·X(6) + X(1)) / ...  [least-squares USL fit]
κ_base = (X(1) − X(6) − α·(X(1)−X(6)·5)) / ...
CG_mean = κ_base / κ_eff_measured
```

In practice, H2AI issues a `POST /calibrate` request and publishes a `CalibrationStartedEvent`. The autonomic loop runs the structured task battery, fits the parameters, computes `N_max` and `θ_coord`, and publishes `CalibrationCompletedEvent` with the results.

**Recalibration triggers:**

| Trigger | Reason |
|---------|--------|
| New LLM adapter registered | κ_base changes with model capability |
| Agent count changes by ≥ 2 | CG_mean shifts with new temperature distribution |
| ZeroSurvivalEvent rate > 10% over 1h window | κ_eff is higher than calibrated; system is past N_max |
| ADR corpus grows by ≥ 5 ADRs | J_eff distribution shifts; gate thresholds should be re-verified |

> **Script cross-reference:**
> - `validate_math.py` — `CALIBRATION_TABLE`: the single source of truth for all three layer constants; "Calibration Reference Table — Full Cross-Check" section asserts κ_eff and N_max for every row, and asserts algebraic equivalence of both N_max forms
> - `simulate_usl.py` — `LAYERS`: must contain the same (α, κ_base, CG_mean) triples as `CALIBRATION_TABLE`; used in Plots 1 and 2. **These two constants must stay in sync with this table.**

---

## 4. Runtime Safety Constraints

The following constraints are enforced by the orchestrator before any topology is committed:

| Constraint | Formal condition | Enforcement point |
|------------|-----------------|-------------------|
| J_eff gate | `J_eff ≥ 0.4` | Task submission → `ContextUnderflowError` if violated |
| N ceiling  | `N ≤ N_max` | Explorer spawn rejection |
| Multiplication condition | All three Prop 3 conditions | Explorer provisioning |
| Coordination threshold  | `CG(i,j) ≥ θ_coord` for all adjacent pairs | Topology construction |
| Merge safety | `max(c_i) ≤ 0.85` for CrdtSemilattice | Merge authority |
| Idempotency | TTL-bounded request-ID deduplication | Adapter layer |

> **Script cross-reference:** `J_EFF_GATE = 0.4` appears in both scripts. `BFT_THRESHOLD = 0.85` appears in `validate_math.py`. These constants **must** match this table.

---

## 5. Event Vocabulary

Every runtime state transition emits an immutable event to NATS JetStream. The 14-event vocabulary is the operational realisation of the mathematical model:

| Event | Mathematical meaning |
|-------|---------------------|
| `CalibrationStartedEvent` | System begins fitting α, κ_base, CG_mean |
| `CalibrationCompletedEvent` | Parameters committed; N_max and θ_coord available |
| `TaskSubmittedEvent` | J_eff measured against ADR corpus |
| `ContextUnderflowError` | J_eff < 0.4; Def 10 gate triggered |
| `MultiplicationConditionFailedEvent` | Prop 3 violated; topology cannot multiply |
| `ExplorerSpawnedEvent` | Agent added to topology; N checked against N_max |
| `ProposalGeneratedEvent` | Agent output ready for Auditor |
| `BranchPrunedEvent` | Auditor applied a constraint from ADR corpus |
| `MergeCompletedEvent` | CRDT semilattice or BFT consensus merge completed |
| `ZeroSurvivalEvent` | All branches pruned; MAPE-K retry triggered |
| `TopologyRetryEvent` | MAPE-K loop adjusting N and τ |
| `SemilatticeCompiledEvent` | Final merge committed; Prop 5 verified |
| `TaskCompletedEvent` | Task delivered to caller |
| `TaskFailedEvent` | Non-recoverable failure after retry exhaustion |

---

## 6. Citations

1. Gunther, N. J. (1993). *A Simple Capacity Model of Massively Parallel Transaction Systems.* CMG Conference Proceedings.
2. Papamarcos, M. S. & Patel, J. H. (1984). *A Low-Overhead Coherence Solution for Multiprocessors with Private Cache Memories.* ISCA '84, pp. 348–354. DOI: 10.1145/800015.808204.
3. Condorcet, M. J. A. N. de (1785). *Essai sur l'Application de l'Analyse à la Probabilité des Décisions Rendues à la Pluralité des Voix.*
4. Hong, L. & Page, S. E. (2004). *Groups of Diverse Problem Solvers Can Outperform Groups of High-Ability Problem Solvers.* PNAS, 101(46), 16385–16389. DOI: 10.1073/pnas.0403723101.
5. Matsutani, S. et al. (2023). *Conway's law, revised from a mathematical viewpoint.* arXiv:2311.10475.
6. Kim, Y. et al. (2025). *Towards a Science of Scaling Agent Systems.* arXiv:2512.08296.
7. Wang, Y. et al. (2026). *OrgAgent: Organize Your Multi-Agent System like a Company.* arXiv:2604.01020.
8. Dunbar, R. I. M. (1992). *Neocortex Size as a Constraint on Group Size in Primates.* Journal of Human Evolution, 22(6), 469–493.
