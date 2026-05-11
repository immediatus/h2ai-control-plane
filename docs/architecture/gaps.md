# H2AI Gaps — Research and Engineering Agenda

This document is the actionable companion to [`research-state.md`](research-state.md). Every gap
is a falsifiable question with a concrete research or engineering path. Where previous editions
listed status only, this edition adds: literature grounding, innovative solution design,
mathematical improvement, and simulation protocol for every open gap.

---

## Navigation

| Section | What it covers |
|---|---|
| [Problem Space Map](#problem-space-map) | At-a-glance status and severity for all gaps |
| [Innovations Roadmap](#innovations-roadmap) | Cross-cutting solutions that close multiple gaps simultaneously |
| [Group A — Core Thesis](#brainstorm-group-a--core-thesis-validity) | Does the fundamental approach work and beat its competitors? |
| [Group B — Math Apparatus](#brainstorm-group-b--mathematical-formula-validity) | Are the formulas principled or arbitrary? |
| [Group D — Infrastructure](#brainstorm-group-d--infrastructure-and-operational-gaps) | Do the inputs to the math arrive correctly? |
| [Group E — Quality Measurement](#brainstorm-group-e--quality-measurement-infrastructure) | Can we measure what we claim to optimise? |
| [New Gaps](#new-gaps-from-2026-05-13-review) | Newly identified from critical review and arXiv research |
| [Shared Infrastructure](#shared-infrastructure-required-for-group-a) | Pre-work that blocks Group A experiments |

---

## Problem Space Map

| Gap | Status | Severity | Innovation opportunity |
|---|---|---|---|
| **GAP-A1 Self-MoA vs. multi-family routing** | 🟡 PARTIAL | **Critical** | H2-P vs. B3 experiment runnable today |
| **GAP-A2 USL N_max vs. quality curve** | 🔴 OPEN | **Critical** | Replace USL with N_IT as primary sizer; USL → cost cap only |
| **GAP-A6 Self-MoA as direct empirical competitor** | 🔴 OPEN | **Critical** | Structured experiment with constraint compliance as primary metric |
| **GAP-A7 Preference leakage in LlmJudge** *(new)* | 🔴 OPEN | **High** | Cross-family judge rotation via existing adapter factory |
| **GAP-B1 β_eff functional form** | 🔴 OPEN | Medium | First-principles derivation now available — unify with context-aware formula |
| GAP-B3 Attribution self-referential | 🟡 PARTIAL | Medium | Conformal prediction once oracle data exists |
| **GAP-B5 p_mean / rho_mean no derivation** *(new)* | 🔴 OPEN | **High** | Derivation now possible from CJT + Hamming geometry |
| **GAP-D1 Calibration measures API latency, not coordination cost** | 🔴 OPEN | **Critical** | Conflict-count β signal from existing verifier output |
| GAP-D2 Compound task cost unconstrained | 🔴 OPEN | Low | Complexity bandit probe |
| GAP-D3 Bootstrap calibration | 🔴 OPEN | Low | Built-in synthetic prompt set |
| GAP-E1 Oracle integration | 🟡 PARTIAL | Blocking | Domain-specific test suites remaining |
| **GAP-E2 Talagrand feedback loop** | 🔴 OPEN | Medium | τ-spread KL update rule |

**Severity key** — Critical: threatens core thesis validity; High: corrupts math inputs or silently disables documented features; Medium: degrades confidence in results; Low: operational or presentation issue.

---

## Innovations Roadmap

Three cross-cutting innovations that each close multiple gaps without requiring new infrastructure.
Implement these before running any Group A experiments — the experiments will produce better-
grounded data if the math inputs are correct.

### INNOVATION-2 — Conflict-Count β₀ (replaces API-latency β)

**Closes:** GAP-D1.  
**Cost:** 1 week to extend calibration harness.  
**Why now:** The constraint verifier already runs on every proposal in Phase B calibration. Counting
constraint violations per proposal pair is free — the data already flows through.

**Current state (wrong signal):**

```
β₀ = (z_M - z_2 × (M-1)) / ((M-1)(M-2))   where z = wall-clock time
```

Fast API (local LLM) → small z → small β₀ → large N_max.  
This is backwards: local single-model deployments produce the most correlated proposals and need
the *smallest* N_max, not the largest.

**Proposed state (principled signal):**

USL's β is defined as the cost of N(N-1) pairwise coherency checks. In H2AI, a coherency check
is a constraint conflict resolution. The coordination cost between agents i and j is proportional
to the number of constraints where they disagree.

```
conflict_rate(M) = (1/M(M-1)) × Σ_{i≠j} |constraints_violated_by_i XOR violated_by_j| / |corpus|

β_conflict = (conflict_rate(M) - conflict_rate(2)) / ((M-1)(M-2))
```

Two β values in parallel:
- `beta_latency` — existing timing β (useful for latency estimation only)
- `beta_quality` — new conflict-count β (drives N_max for quality bounding)

N_max uses `beta_quality`. Latency prediction uses `beta_latency`. These are separate concerns and
should never share the same estimate.

**First-principles derivation of β_eff:**

Given `beta_quality`, the β_eff formula now has a derivation:

```
β_eff = β_quality × (1 - CG_mean)
```

Interpretation: `β_quality` is the conflict cost per pair per unit CG distance.
`(1 - CG_mean)` is the expected CG distance between any two adapters in the pool.
Product: expected conflict cost per adapter pair in the actual pool.

This is not a heuristic — it follows from the definition if `conflict_rate ∝ (1 - CG)`. That
proportionality is an empirical claim to verify, but it is now falsifiable: measure
`conflict_rate` and `CG_mean` independently and fit the linear coefficient.

**Simulation:**

```python
# Validate: does constraint conflict rate scale linearly with (1 - CG_mean)?
# Use synthetic binary satisfaction vectors
import numpy as np

def conflict_rate(profiles):
    """profiles: (N, K) binary array where profiles[i,k] = agent i satisfies constraint k"""
    N, K = profiles.shape
    conflicts = 0
    for i in range(N):
        for j in range(i+1, N):
            conflicts += np.sum(profiles[i] != profiles[j]) / K
    return conflicts / (N * (N-1) / 2)

def hamming_cg(profiles):
    N, K = profiles.shape
    agree = 0
    for i in range(N):
        for j in range(i+1, N):
            agree += np.sum(profiles[i] == profiles[j]) / K
    return agree / (N * (N-1) / 2)

# Sweep CG_mean by varying cross-profile overlap probability p_agree
results = []
for p_agree in np.linspace(0.3, 0.95, 15):
    profiles = (np.random.rand(6, 100) < p_agree).astype(int)
    # Introduce controlled disagreement: flip with prob (1-p_agree)
    for i in range(1, 6):
        mask = np.random.rand(100) < (1 - p_agree)
        profiles[i] = np.where(mask, 1 - profiles[0], profiles[0])
    cg = hamming_cg(profiles)
    cr = conflict_rate(profiles)
    results.append((cg, cr))
    print(f"CG_mean={cg:.3f}, conflict_rate={cr:.3f}, 1-CG={1-cg:.3f}")

# Fit: conflict_rate ~ β₀ × (1 - CG_mean)
cgs, crs = zip(*results)
from numpy.polynomial import polynomial as P
coef = np.polyfit([1 - c for c in cgs], crs, 1)
print(f"Fit: conflict_rate = {coef[0]:.3f} × (1 - CG) + {coef[1]:.4f}")
print(f"If R² > 0.95, the linear form is validated.")
```

### INNOVATION-4 — N_IT as Primary Sizer; USL as Cost Cap

**Closes:** GAP-A2 (reframes rather than abandons USL).  
**Cost:** 1 week (routing logic change; no new math).  
**Why now:** `n_it_optimal` is already implemented in math.md §5.1 and matches `condorcet_n_optimal`
within ±1 for ρ ∈ [0.3, 0.95]. The information-theoretic framing is self-contained and valid
without the USL domain-transfer assumption. USL's role should be cost cap, not quality target.

**Current state:**

```
target N = n_optimal (Condorcet, maximises marginal Q gain)
ceiling N = N_max (USL, throughput model applied to quality)
```

Both are "quality" functions. USL's quality claim is unvalidated (no arXiv paper applies USL to
LLM agent ensembles — confirmed by our search). n_it_optimal is derived from independent
information theory and is correct on its own terms.

**Proposed state:**

```
target N = n_it_optimal = ceil(log(0.5) / log(1 - ρ))   [information-theoretic target]
ceiling N = N_max_USL                                     [cost ceiling, not quality target]
final N   = min(target_N, ceiling_N, calibration_max_ensemble_size)
```

The USL ceiling prevents runaway token cost when ρ is underestimated (which it currently is due
to the proxy chain). It is honest as a cost heuristic. It is not honest as a quality predictor.
Documents and code should describe N_max as a cost cap explicitly.

**Why information-theoretic N is sound:**

Marginal information gain of agent k in an N-agent ensemble with pairwise correlation ρ:

```
I_k = H(X) × (1 - ρ)^(k-1)     [geometric decay due to shared information]
```

Summing: total information = H(X) × (1 - (1-ρ)^N) / ρ

N_IT = min N: (1-ρ)^(N-1) < 0.5  →  N_IT = ceil(log(2) / log(1/(1-ρ)))

This is the point where the marginal information drops below half the per-agent entropy. Adding
more agents beyond this gives diminishing returns regardless of cost.

**Simulation — N_IT vs N_max across ρ values:**

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

alpha = 0.12
beta0 = 0.039

def n_max_usl(cg_mean):
    beta_eff = beta0 * (1 - cg_mean)
    return max(1, round(np.sqrt((1 - alpha) / beta_eff)))

def n_it_optimal(rho):
    if rho <= 0: return 100
    return max(1, int(np.ceil(np.log(0.5) / np.log(max(1e-9, 1 - rho)))))

rho_values = np.linspace(0.05, 0.90, 50)
cg_values  = 1 - rho_values    # rho_proxy = 1 - CG_mean

n_usl = [n_max_usl(cg) for cg in cg_values]
n_it  = [n_it_optimal(rho) for rho in rho_values]

print("rho | N_IT | N_max_USL | gap")
for rho, n_i, n_u in zip(rho_values[::5], n_it[::5], n_usl[::5]):
    print(f"{rho:.2f} | {n_i:4d} | {n_u:9d} | {n_u - n_i:+4d}")
# Expected: N_max_USL >> N_IT for low ρ; USL is too permissive
```

---

### INNOVATION-5 — Structured Self-MoA Experiment Protocol

**Closes:** GAP-A6.  
**Cost:** 1 week (single-model, runnable today on devcontainer).  
**Why now:** The H2-P vs. B3 comparison does not require a second model family. It is the minimum
viable falsification of H2AI's core claim and can run immediately.

**Experiment design (phases):**

**Phase 0 — Baseline (B0):** Single shot, no retries.
**Phase B3 — Self-MoA:** Same model, τ ∈ {0.2, 0.7, 1.0}, 3 proposals, majority-vote on
  verification score, no constraint enforcement loop, no Phase 0 decomposition.
**Phase H2-P — H2AI Precision:** Same model, τ-spread, Phase 0 decomposition, MAPE-K constraint
  enforcement, phase 3.5 rubric scoring, full pipeline.

Primary metric: **constraint compliance rate** (fraction of constraint checks passed in oracle
evaluation — NOT just internal verifier score). Secondary: oracle pass rate on code tasks.

Key prediction: H2-P > B3 on 5+ constraint tasks even when the model is the same, because:
1. Phase 0 decomposition focuses each slot on a distinct constraint domain
2. MAPE-K enforcement iteratively raises constraint satisfaction
3. Self-MoA (B3) has no enforcement loop — it produces higher diversity but no constraint gate

If H2-P ≤ B3 on constraint-heavy tasks, the MAPE-K enforcement adds nothing above temperature
diversity and the core claim fails. Document that outcome and pivot to positioning H2AI as
"structured Self-MoA with better observability."

**Task selection:** S2 HumanEval set stratified by constraint count:
- Tier 1: 1–2 constraints (simple)
- Tier 2: 3–5 constraints (moderate)
- Tier 3: 6+ constraints (complex)

Hypothesis: H2-P advantage over B3 should grow with tier (more constraints = more MAPE-K value).
If advantage is flat across tiers, the hypothesis is wrong.

---

## Foundational Framing — Every Problem Is a Team Epistemology Problem

Any non-trivial problem is a **team knowledge acquisition problem**: the team must discover what is
true about the problem domain, resolve contradictions between what different team members believe,
and produce a justified output that survives contact with reality. The solution is not a pipeline
— it is a **graph of thinking, decisions, and executions**, with loops wherever knowledge needs
to be refined.

### The knowledge graph

```
Nodes  : beliefs  — {claim, evidence, assumptions, scope, confidence}
Edges  : support  (B strengthens A)
         contradiction  (B and A cannot both be true)
         derivation     (B follows from A by rule R)
         grounding      (oracle/test connects A to external reality)

Loops  :
  coherence   → resolve contradictions between beliefs until none remain
  coverage    → ensure every required knowledge dimension has a justified belief
  grounding   → connect load-bearing beliefs to external reality
  revision    → update beliefs when new evidence contradicts old ones
```

### How H2AI phases map to epistemic operations

| Phase | Epistemic operation |
|---|---|
| Task decomposition (Phase 0) | Epistemic division of labor — assign knowledge responsibilities |
| TAO inner loop | Hypothesis exploration — form initial beliefs |
| Phase 2 topology | Determine how many independent knowledge contributors the budget allows |
| Phase 3.5 verification | Coherence test — do beliefs satisfy the constraint axioms? |
| Phase 4 audit | Final coherence gate before a belief is accepted as output |
| Phase 5a synthesis | Belief integration — construct the most coherent view across contributors |
| MAPE-K retry | Belief revision — update under new evidence from failed coherence tests |
| Phase 6 oracle (GAP-E1) | **Grounding** — connect beliefs to external reality |
| Calibration | Update meta-beliefs about team epistemic capabilities |

### The epistemological traditions each gap violates

| Gap | Epistemic violation |
|---|---|
| GAP-A6: Self-MoA as competitor | Core diversity premise is empirically untested |
| GAP-E1: no oracle | Epistemic closure — internally coherent but ungrounded |
| GAP-B3: attribution without oracle | Cannot distinguish confident-and-correct from confident-and-wrong |
| GAP-D1: calibration measures wrong signal | β₀ captures API speed, not coordination cost |
| GAP-B5: rho_mean derivation | Convention `rho_mean = 1 − CG_mean` lacks derivation; empirical ρ EMA now live |

### Stopping criteria

| Loop | Current criterion | Principled criterion | Gap |
|---|---|---|---|
| TAO inner | `agent_max_tool_iterations` (budget) | No productive hypothesis extensions remain | Budget is proxy for epistemic exhaustion |
| MAPE-K retry | Proposals satisfy threshold OR retries exhausted; ZeroSurvival + is_closed() gate | Coherent closure: no active constraint violated, no domain uncovered | Quality threshold is rubric-coherent, not oracle-grounded |
| Calibration | Startup-automatic + POST /calibrate | Confidence intervals narrow enough for decision quality required | GAP-D1: calibration measures latency, not epistemic conflict cost |
| Oracle grounding | Phase 6 wired; OracleWorker + OracleAccumulator live | All load-bearing beliefs grounded in at least one oracle test | GAP-E1: domain-specific test suites remaining |

---

## Brainstorm Group A — Core Thesis Validity

---

### GAP-A1: Self-MoA vs. Multi-Family — Does Diversity Matter? 🟡 PARTIAL

**Status: PARTIAL** — TCC parameters are unfit priors; quality comparison not yet measured.

**Literature grounding.**
arXiv 2502.00674 (Li et al., 2025) — *"Rethinking Mixture-of-Agents: Is Mixing Different LLMs
Beneficial?"* — proposes Self-MoA: aggregating outputs from a single top-performing model with
temperature variation. Finds Self-MoA matches or outperforms cross-model MoA on most benchmarks.
The argument: mixing weaker models dilutes the strong model's signal.

arXiv 2512.17630 (Elgabry & Hamdi, 2025) — *"Confidence-Credibility Aware Weighted Ensembles"*
— explicitly invokes CJT and confirms: CJT advantage holds only when error diversity is actively
maintained. Minimising parameter convergence (distinct architectures) is essential.

arXiv 2411.01539 (Bradley, 2024) — *"LLMs and the Madness of Crowds"* — LLM errors are
systematically correlated across architecturally similar models. Naive majority-vote ensembles
reinforce errors when models share the same failure mode.

**Open.**
TCC parameters are unfit priors. The smoke test measured latency, not quality. `shadow_mode =
false` means unfitted priors are live in production. The 2×2 cross-family experiment requires a
second model family not currently available.

**Innovative solution (available today).**
Run INNOVATION-5 (H2-P vs. B3 experiment protocol). H2-P is the minimum viable falsification:
same model, τ-spread, Phase 0 decomposition, full constraint enforcement loop. This can run on
the devcontainer today. The result either validates H2AI's Phase 0 + MAPE-K claim or reveals that
constraint enforcement adds nothing above temperature diversity.

**Falsification condition.**
If H2-P ≤ B3 on Tier 3 tasks (6+ constraints) with oracle pass rate as the signal (not internal
verifier score), the Coverage routing adds cost without benefit and should be replaced by Precision
routing universally.

---

### GAP-A2: USL N_max vs. Actual Quality Curve 🔴 OPEN — **Critical**

**Gap statement.**
`N_max = round(√((1−α)/β_eff))` is derived from USL's throughput model by setting `dX/dN = 0`.
USL models CPU/network throughput under parallelism — not output quality. No published arXiv paper
applies USL to LLM multi-agent quality ceiling. This is confirmed by our search: zero results for
`abs:"universal scalability law" AND abs:agent`. The domain transfer assumption is unvalidated.

Additionally, USL's ceiling and the Condorcet n_optimal serve different purposes: USL caps cost;
Condorcet maximises quality-per-agent. Using a cost model as a quality predictor can cause
over-sizing (high ρ → USL gives high N_max because β is suppressed, but Condorcet gives low N
because diversity is poor) or under-sizing (low ρ → large N_IT but USL may cap lower than optimal).

**Literature grounding.**
arXiv 0808.1431 (Gunther, 2008) — foundational USL derivation. β is defined as the "coherency
cost" — serialisation overhead per N(N-1) pairwise checks. Applicable to compute platforms;
domain transfer to quality requires explicit argument.

arXiv 2006.04969 (Hamann & Reina, 2020) — applies USL and Amdahl's law to robot swarms: the
closest bridge to autonomous agent systems. Finds USL describes swarm throughput; quality metrics
(task success rate) follow different scaling laws.

arXiv 2509.19489 (Nowak, 2025) — derives optimal compute allocation for self-consistency under
budget B = m×n: optimum is m,n ∝ √B, not n→∞. Directly constrains the ensemble budget
allocation, independent of USL.

**Innovative solution — INNOVATION-4: N_IT as primary sizer.**

Promote `n_it_optimal` to primary N recommendation. Demote N_max_USL to cost cap. The
information-theoretic formula has a self-contained derivation (see INNOVATION-4 above) and does
not require the USL domain-transfer assumption. In code: rename or document N_max as
`n_max_cost_ceiling` and adjust the planning logic:

```rust
let n_target = calibration.n_it_optimal(rho_mean);   // info-theoretic target
let n_max    = calibration.n_max();                   // USL cost ceiling
let n_final  = n_target.min(n_max).min(cfg.calibration_max_ensemble_size);
```

Document USL explicitly as: "N_max is a cost heuristic drawn from throughput engineering. It
prevents runaway token cost but is not a quality predictor. The quality target is n_IT."

**Simulation — validate N_IT vs empirical quality curve:**

```python
# Monte Carlo: quality as a function of N for given p and rho
import numpy as np
from scipy.stats import binom

def monte_carlo_ensemble_quality(n, p, rho, n_tasks=10000):
    """Simulate majority-vote ensemble quality with correlated errors."""
    # Generate correlated binary outcomes using Gaussian copula
    from scipy.stats import norm
    # Cholesky decomposition of correlation matrix
    corr_matrix = rho * np.ones((n, n)) + (1 - rho) * np.eye(n)
    L = np.linalg.cholesky(corr_matrix)
    u = norm.cdf(np.random.randn(n_tasks, n) @ L.T)
    correct = (u < p).astype(int)
    majority_correct = (correct.sum(axis=1) > n // 2).mean()
    return majority_correct

p, rho = 0.70, 0.30
print("N | Q_condorcet | Q_monte_carlo | N_IT | N_max_USL")
for n in [1, 2, 3, 5, 7, 9]:
    q_cjt = p + (sum(binom.pmf(k, n, p) for k in range(n//2+1, n+1)) - p) * (1 - rho)
    q_mc  = monte_carlo_ensemble_quality(n, p, rho)
    n_it  = max(1, int(np.ceil(np.log(0.5) / np.log(max(1e-9, 1-rho)))))
    n_usl = max(1, round(np.sqrt(0.88 / (0.039 * (1 - (1-rho))))))
    print(f"{n} | {q_cjt:.3f}       | {q_mc:.3f}         | {n_it}    | {n_usl}")
```

---

### GAP-A6: Self-MoA Is a Direct Empirical Competitor 🔴 OPEN — **Critical**

**Gap statement.**
arXiv 2502.00674 (Li et al., 2025) — Self-MoA beats multi-family MoA by 6.6% on AlpacaEval 2.0.
H2AI's full machinery must beat a single strong model at multiple temperatures to justify its
existence. The Coverage quadrant (cross-family committee) is the architectural bet against
Self-MoA; it has not been measured.

**Literature grounding.**
arXiv 2502.00674 — Self-MoA paper; primary threat.

arXiv 2505.24442 (Xie et al., 2025) — *"RMoA: Optimising MoA through Diversity Maximisation and
Residual Compensation"* — finds explicit diversity maximisation in agent selection is "the single
biggest lever for MoA quality." Validates H2AI's Phase 0 decomposition as the right intervention;
now needs empirical validation against B3 baseline.

arXiv 2601.16715 — *"Dynamic Expert-Guided Model Averaging"* — LLM-guided ensemble weighting
outperforms uniform averaging when the LLM's structural knowledge is partially correct. Validates
Phase 0 mandate-based slot selection.

arXiv 2503.03535 — *"Trade-offs in Ensembling, Merging, and Routing"* — systematic comparison:
ensembling dominates on distribution-shift tasks; routing wins in-distribution. Constraint-
compliance tasks under novel specifications are distribution-shift scenarios → ensemble approach
is architecturally correct for H2AI's target domain.

**H2AI's specific claim and where it can win.**
The Precision quadrant (τ-spread within one family) is architecturally aligned with Self-MoA.
The Coverage quadrant (cross-family) is the bet against it. H2AI's additional bet: the MAPE-K
enforcement loop adds constraint compliance above what any temperature-spread ensemble can achieve
without an enforcement mechanism.

**Innovative solution — INNOVATION-5: Structured experiment protocol.**

See INNOVATION-5 above. The key insight from the literature: H2AI should not try to beat Self-MoA
on general benchmarks. The correct metric is **constraint compliance rate on multi-constraint
tasks** — tasks where Self-MoA has no enforcement mechanism and MAPE-K has structural advantage.

**Concrete falsification table:**

| Condition | Task | Expected result if H2AI hypothesis holds |
|---|---|---|
| H2-P vs B3 | Tier 1 (1-2 constraints) | B3 ≈ H2-P (enforcement loop rarely fires) |
| H2-P vs B3 | Tier 3 (6+ constraints) | H2-P > B3 (enforcement loop provides value) |
| H2-C vs B3 | Any tier | Unmeasurable until second adapter family available |

If the Tier 3 advantage does not materialise, H2AI's value is observability and governance (audit
trail, constraint provenance) rather than quality improvement — which is still a valid product
position but requires rewritten documentation.

---

## New Gaps From 2026-05-13 Review

---

### GAP-A7: Preference Leakage in LlmJudge 🔴 OPEN — **High**

**Gap statement.**
When the constraint evaluation adapter (LlmJudge in Phase 3.5 and Phase 4) belongs to the same
LLM family as the explorer adapters, the judge systematically favours outputs that match its
family's stylistic patterns. This is not a neutral evaluation — it is an in-family affinity bias
that inflates verification scores for outputs that echo the judge's own training distribution.

**Literature grounding.**
arXiv 2502.01534 (Li et al., 2025) — *"Preference Leakage: A Contamination Problem in LLM-as-a-
Judge"* — demonstrates that when the same LLM family generates synthetic training data AND judges
it, the judge systematically favours the model's own output style. Feedback loop: the judge's
high scores reinforce generator outputs that look like the judge's own style, creating a
self-amplifying bias.

arXiv 2410.13341 (Dorner et al., 2024) — *"LLM as Judge Won't Beat Twice the Data"* — proves
that LLM judge self-preferencing bias is irreducible. Debiasing tools reduce but cannot eliminate
it. Pairwise evaluation (arXiv 2504.14716) is less biased than pointwise but introduces
Condorcet cycles at scale.

**Current state.**
`VerifierExplorerFamilyConflict` is a hard gate that prevents the explorer pool and the
*verification adapter* from sharing a family when `family_constraint = "require_diverse"`. This partially
addresses the issue for Phase 3.5. However:

1. The Phase 4 shadow auditor's family is enforced separately (env var family guard in main.rs).
2. There is no check on whether the Phase 4 PRIMARY auditor shares a family with the explorer pool.
3. The `Disabled` variant of `family_constraint` disables all family conflict detection.

**Innovative solution — Cross-family judge rotation.**

The adapter factory already supports multiple families. Add a `constraint_judge` config section:

```toml
[constraint_judge]
# "auto" = select adapter from non-majority explorer family; falls back to any available
# "explicit" = use named provider (overrides auto)
family_strategy = "auto"
```

Implementation in `engine.rs`, at Phase 3.5 adapter selection:
1. Count explorer adapter families in the current wave.
2. Select the verifier adapter from the factory excluding the majority family.
3. If only one family is available (single-model deployment), document the limitation in the
   `VerifierExplorerFamilyConflict` warning rather than silently proceeding.

This re-uses existing infrastructure (adapter factory, family guards) and requires no new
concepts.

**Falsification condition.**
Compare verification scores between (a) same-family judge and (b) cross-family judge on a labeled
task set where oracle pass rate is known. If cross-family judge Spearman correlation with oracle is
significantly higher (p < 0.05) than same-family, the preference leakage is measurable.

---

## Brainstorm Group B — Mathematical Formula Validity

---

### GAP-B1: β_eff Functional Form Has No Derivation 🔴 OPEN — Medium

**Gap statement.**
`β_eff = β₀ × (1 − CG_mean)` is stated as a design choice. The linear form is unvalidated.

**Literature grounding.**
arXiv 0808.1431 (Gunther, 2008) — USL β is defined as the per-pair coherency synchronisation
overhead. In H2AI terms, this is the per-adapter-pair constraint conflict resolution cost.

**Innovative solution — First-principles derivation (resolves the gap conceptually).**

In USL, β is the fraction of time spent on N(N-1) pairwise coherency checks. In H2AI's merge:

- Expected constraint conflicts between adapters i and j ∝ (1 - CG(i,j)) × |corpus|
- Mean expected conflicts ∝ (1 - CG_mean)
- Conflict resolution cost ∝ n_conflicts (approximately linear — one synthesis token per conflict)

Therefore: `β_eff ∝ β₀ × (1 - CG_mean)` is a first-principles consequence if conflict resolution
cost is linear in conflict count. The linear form is correct under this assumption.

The assumption is falsifiable: if conflict resolution cost is super-linear (e.g. due to attention
degradation in a long synthesis context), the formula needs a higher-order term.

**Improved formula unifying with context-aware term (from math.md §2.3):**

```
β_eff(N) = β₀ × (1 - CG_mean) × (1 + γ × fill(N))
fill(N)  = min(1, N × proposal_tokens / max_tokens)
```

This is already computed in `n_max_context_aware` but is not used in the primary `n_max()`
function. Unify them: make the attention term the standard formula, not a contextual variant.

**Empirical validation with INNOVATION-2 data:**
When the conflict-count β (INNOVATION-2) is available, regress `conflict_rate` against
`(1 - CG_mean)` and `(1 - CG_mean) × fill(N)`. If the bivariate R² > R² of the univariate fit,
the attention term has explanatory power.

---

### GAP-B3: Attribution Formula Is Self-Referential 🟡 PARTIAL

Oracle-grounded calibration requires GAP-E1 data. Once available, apply conformal prediction:

```
conformal_margin(α) = quantile(|q_confidence - q_oracle|, 1-α)  over calibration split
output: q_confidence ± conformal_margin(0.10)   [90% coverage guarantee]
```

arXiv 2410.11594 (Wagner et al., 2024) — *"Black-box Uncertainty Quantification for LLM-as-a-
Judge"* — sampling-based confidence intervals for LLM judge scores without white-box access.
Applicable to H2AI's Phase 3.5 verifier score intervals.

---

### GAP-B5: rho_mean Has No Derivation 🔴 OPEN — **High**

**Gap statement.**
`rho_mean = 1 − CG_mean` is used as the correlation correction in `Q(N, p, ρ)`. No derivation.
The formula implies CG_mean = 0 (zero constraint agreement) → ρ = 1.0 (fully correlated).
This is backwards: low CG_mean means agents disagree on constraints, which should indicate *less*
error correlation, not more.

**Analysis of the implicit logic.**

The intended interpretation: CG_mean measures *similarity* of behavioural profiles. High
similarity → high correlation. So `ρ = CG_mean` would be the natural proxy.

The actual formula `ρ = 1 - CG_mean` inverts this: high similarity → low ρ. The implicit
reasoning may be: "if agents agree (high CG), they're diverse in other ways, so ρ is low" — but
this is neither stated nor justified.

**Alternative derivation from Hamming geometry:**

If two agents have CG(i,j) = c (constraint agreement fraction), then:
- The fraction of constraints where they disagree = 1 - c
- Disagreement means independent error patterns on those constraints
- Agreement means correlated error patterns on those constraints
- If errors only occur on disagreed constraints: ρ ∝ c (agreement)

This gives `ρ_mean = CG_mean`, not `1 - CG_mean`.

However: agreement on constraint profiles doesn't mean agreement on *which wrong answer* to
give. Two agents can agree on satisfying constraint k but disagree on constraint l — yet both
hallucinate the same wrong entity. The SRANI CFI mechanism is precisely designed to detect this.

**Pragmatic conclusion:** Neither `ρ = CG_mean` nor `ρ = 1 - CG_mean` is derivable from first
principles without additional assumptions. Online ρ_EMA is now live (`rho_ema.rs`) and replaces
the proxy once 30 task observations accumulate. Until that threshold is reached, the system uses:

> "Operational convention: low CG (diverse constraint profiles) is assumed to indicate lower error
> correlation. This assumes error patterns track constraint specialisation. The assumption is
> unvalidated and replaced by empirical ρ_EMA once 30 task observations exist."

**Simulation to reveal the assumption sensitivity:**

```python
import numpy as np

def q_condorcet(n, p, rho):
    from scipy.stats import binom
    q_ind = sum(binom.pmf(k, n, p) for k in range(n//2 + 1, n+1))
    return p + (q_ind - p) * (1 - rho)

# Show sensitivity of Q to ρ=CG vs ρ=1-CG for CG_mean=0.7
cg = 0.70
p  = 0.5 + cg / 2   # = 0.85
print("Formula | rho | Q(N=5)")
for rho_formula, label in [(cg, "rho=CG"), (1-cg, "rho=1-CG")]:
    q = q_condorcet(5, p, rho_formula)
    print(f"{label:10s} | {rho_formula:.2f} | {q:.4f}")
# If the two give very different Q, the formula choice matters significantly
```

---

## Brainstorm Group D — Infrastructure and Operational Gaps

---

### GAP-D1: Calibration Measures API Round-Trip Latency, Not Coordination Cost 🔴 OPEN — **Critical**

**Gap statement.**
Phase A and Phase B of the calibration harness measure wall-clock time to fit α and β₀. USL's β
is a coherency cost — serialisation overhead of N(N-1) pairwise coherency checks. H2AI's β₀ is
computed from elapsed time divided by T₁: this measures API round-trip latency, not constraint
conflict resolution cost. The measurement is systematically inverted:

- Fast local LLM (50ms) → small T₁ → small β₀ → large N_max → too permissive for single-model
- Slow cloud LLM (3s) → large T₁ → large β₀ → small N_max → too conservative for diverse pool

**Literature grounding.**
arXiv 0808.1431 (Gunther, 2008) — β in USL is defined as the coherency synchronisation cost
fraction per adapter pair, not total wall-clock time. The fit formula requires latency at N=2 and
N=M relative to a reference — but the reference must be the *serialisation cost*, not the
generation cost. In H2AI, generation cost (API time) dominates; serialisation cost (CG computation
+ merge context build) is negligible in comparison.

**Innovative solution — INNOVATION-2: Conflict-count β₀.**

See INNOVATION-2 above for full derivation and simulation. Implementation:

During Phase B calibration, after all M adapters respond, run the constraint verifier on each
proposal and record:

```rust
struct CalibrationConflictSample {
    n_adapters: u32,
    pairwise_conflict_rate: f64,   // mean |violated_by_i XOR violated_by_j| / |corpus| over pairs
}
```

Fit `beta_quality`:

```rust
beta_quality = (conflict_rate_M - conflict_rate_2) / ((M - 1) * (M - 2)) as f64
```

Add to `CalibrationCompletedEvent`:
```rust
pub beta_latency: f64,   // existing timing β (latency estimation only)
pub beta_quality: f64,   // new conflict-count β (quality N_max driver)
```

The planning logic uses `beta_quality` for N_max; the operator sees both values. Document:
"beta_latency estimates synthesis wall-clock time; beta_quality estimates constraint coordination
overhead and drives ensemble sizing."

**Online β update via CFI EMA:**

SRANI's CFI is an online proxy for inter-proposal correlated fabrication. High CFI implies high
coordination cost (proposals share fabricated entities → synthesis must reconcile them).

```rust
// In tasks.rs, after each task completes:
if let Some(cfi) = output.srani_cfi {
    calibration.update_beta_from_cfi(cfi, alpha = 0.05);
}

// In EnsembleCalibration:
fn update_beta_from_cfi(&mut self, cfi: f64, alpha: f64) {
    // CFI is a normalised conflict signal [0,1]
    // β_quality should be proportional to expected conflict rate
    let cfi_beta_proxy = self.beta_quality_prior * (1.0 + cfi);
    self.beta_quality_ema = (1.0 - alpha) * self.beta_quality_ema + alpha * cfi_beta_proxy;
}
```

This provides an online conflict-signal update between explicit calibration runs.

---

### GAP-D2: Compound Task Cost Is Unconstrained 🔴 OPEN — Low

A `CompoundTaskEngine` DAG fires a full wave for each subtask with no pre-execution cost estimate
or operator confirmation gate. Up to 75 LLM calls before synthesis for a 5-subtask compound.

**Research approach.** Complexity probe + bandit routing. Before dispatching ensemble, call a
lightweight adapter (smallest available) to rate subtask complexity 1–5. Route 1–2 to single-
adapter path; 3–5 to full ensemble. The probe cost is 1 small-model call vs. N full-model calls.
Use the existing Thompson Sampling bandit to improve probe accuracy over time.

**Effort estimate.** 1 week cost estimate + SSE event; 2 weeks complexity probe + bandit routing.

---

### GAP-D3: Calibration Bootstrapping Has No Defined Path 🔴 OPEN — Low

New deployment with empty KV returns 503 on every task until calibration runs. No automated path.

**Research approach.** Bootstrap calibration mode with built-in synthetic prompts (code, factual,
reasoning). Mark result as `calibration_source: Bootstrap`. Domain-specific calibration overrides.
Helm chart Job that runs automatically on first install.

**Effort estimate.** 1 week bootstrap mode; 1 week Helm chart integration.

---

## Brainstorm Group E — Quality Measurement Infrastructure

---

### GAP-E1: No Oracle Integration 🟡 PARTIAL — Blocking

**Open.** Domain-specific test suites (code, factual QA, structured output) are the remaining work.

**Innovative opportunity — FUSE-style zero-label verifier ensembling.**
arXiv 2604.18547 (Lee et al., 2026) — *"FUSE: Ensembling Verifiers with Zero Labeled Data"* —
proposes unsupervised ensembling of LLM verifiers using only agreement structure among verifiers.
Applicable to H2AI's cold-start problem: before domain-specific oracles exist, use FUSE-style
inter-verifier agreement to produce a calibrated ensemble score without ground truth labels.

**Oracle priority queue by implementation cost:**

| Oracle type | Tasks covered | Implementation cost |
|---|---|---|
| JSON schema validation | Structured output tasks | 1 day — already have typed deserialisation |
| Cargo test / pytest runner | Code tasks | 3 days — ShellExecutor extension |
| MMLU / TriviaQA lookup | Factual QA | 1 week — reference dataset integration |
| Symbolic verifier (Z3) | Formal reasoning | 2 weeks |
| Human rating | Open-ended writing | Open-ended |

Start with JSON schema validation (zero new dependencies, covers structured output tasks which
are H2AI's primary use case) and cargo test runner (enables the GAP-A6 experiment on HumanEval).

---

### GAP-E2: Talagrand Histogram Has No Feedback Loop 🔴 OPEN — Medium

**Gap statement.**
`TalagrandDiagnostic::from_verification_scores` computes the rank histogram. U-shape = over-
confident (increase τ spread); Λ-shape = under-dispersed (decrease τ spread). The feedback loop
is architecturally described but not implemented: `EnsembleCalibration::tau_spread_factor` is not
updated from Talagrand observations.

**Innovative solution — KL-divergence τ-spread update rule.**

arXiv 2605.07775 (Menet et al., 2026) — *"POETS: Uncertainty-Aware LLM Optimisation via Policy
Ensembles"* — uses Thompson Sampling over KL-regularised reward models for ensemble policy
updates. The KL framework directly applies to Talagrand histogram correction.

Define the Talagrand update rule:

```
KL_flat(H) = KL(H || Uniform)   [divergence from flat histogram]
U_score    = variance(H) / mean(H)   [U-shape detection: high variance relative to mean]
Λ_score    = max(H[middle]) / mean(H[edges])   [Λ-shape: centre mass exceeds edges]

Δτ = η × (U_score - Λ_score)
τ_new = clip(τ_current + Δτ, τ_min, τ_max)
```

Where η (learning rate) is fit from the time to cover one effective Talagrand window (default:
30 verification score observations).

Store `tau_spread_factor` history in `H2AI_CALIBRATION` KV alongside α and β₀. This persists τ
tuning across restarts — the system accumulates calibration from production traffic rather than
resetting on each deployment.

**arXiv precedent:**
Meteorological ensemble calibration (the origin of Talagrand histograms) uses exactly this
feedback: when the rank histogram is non-uniform, the spread parameter of the ensemble is adjusted.
The τ-spread in H2AI is the direct analogue of the ensemble spread parameter in NWP.

**Research question:** Is τ adjustment per-task-domain appropriate? A coding task may have
naturally high verification score variance (pass/fail) while a factual QA task may have low
variance. Per-domain Talagrand calibration requires enough observations per domain.

**Effort estimate.** 1 week for KL update rule implementation; 2 weeks for η tuning on a
representative task distribution.

---

## Gap Priority Matrix

| Gap | Core thesis risk | Implementation cost | Data dependency | Suggested order |
|---|---|---|---|---|
| **INNOVATION-2 Conflict-count β₀** | Critical | 1 week | Calibration runs | **Week 1** |
| **INNOVATION-5 H2-P vs B3 experiment** | Critical | 1 week | None (single-model) | **Week 1** |
| **INNOVATION-4 N_IT as primary sizer** | High | 1 week | None | **Week 2** |
| **GAP-A7 Preference leakage** | High | 1 week | None | **Week 2** |
| **GAP-B5 rho_mean documentation** | Medium | 2 days | None | **Week 2** |
| GAP-E1 Domain-specific oracles | Blocking for A/B | 1–3 weeks | Domain test suites | Session 1 |
| GAP-A1 TCC parameter fitting | Critical | 2 weeks | Oracle quality signal | Session 1 |
| GAP-A6 Full experiment (cross-family) | Critical | Timeline open | Second adapter family | Session 2+ |
| GAP-A2 USL quality curve experiment | High | 2 weeks | Shared task set | Session 2 |
| GAP-D1 Calibration harness extension | Critical | 2 weeks | Calibration runs | Session 3 |
| GAP-E2 Talagrand feedback loop | Medium | 3 weeks | Task runs | Session 4 |
| GAP-B1 β_eff functional form fit | Medium | 2 weeks | Controlled calibration | Session 5 |
| GAP-D3 Bootstrap calibration | Low | 2 weeks | None | Any |
| GAP-D2 Compound task cost | Low | 3 weeks | None | Any |

---

## Shared Infrastructure Required for Group A

Sessions 1 and 2 block on building a shared measurement harness:

1. **Labeled task set.** 100–200 tasks across code (test oracle), factual QA (reference answers),
   and constraint-heavy reasoning. Stratified by constraint count:
   - Tier 1: 1–2 constraints (simple) — validates B0 baseline
   - Tier 2: 3–5 constraints (moderate) — tests enforcement value
   - Tier 3: 6+ constraints (complex) — primary H2-P vs B3 battlefield

2. **JSON schema + cargo test oracle.** Minimum viable oracle for Session 1. Structured output
   tasks have zero-dependency validation. Code tasks need a ShellExecutor extension.

3. **Per-N quality measurement.** The benchmark harness extended to record oracle pass rate per
   adapter, per N value (2, 3, 5, 7, 9), and per task tier.

4. **Pairwise error correlation logging.** Per-adapter binary correct/incorrect logged per task.
   Stored to SQLite or Parquet for offline ρ analysis and ρ_EMA validation (online ρ EMA is now
   live in `rho_ema.rs`; offline logging enables retrospective validation of its convergence).

Building this harness is the pre-work for Session 1 and is the first concrete deliverable before
any gap-resolution experiments begin.
