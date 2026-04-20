# H2AI Math Apparatus

This document is the authoritative reference for the mathematical framework underlying H2AI Control Plane.
It describes the formal definitions used in the codebase, the provenance of each formula,
and the honest limitations of what is and is not measured.

---

## 1. Theoretical Foundation: Condorcet Jury Theorem

**Source:** Condorcet (1785), restated in Nitzan & Paroush (1982), extended with correlation by Ladha (1992).  
**Implemented in:** `crates/h2ai-types/src/physics.rs` — `condorcet_quality()`

**Statement:** Given N agents each independently correct with probability p > 0.5,
the probability the majority is correct exceeds p and converges to 1 as N → ∞.

**Definition 1 — Independent ensemble quality:**

```
Q_ind(N, p) = Σ_{k=⌈N/2⌉+1}^{N} C(N,k) × p^k × (1−p)^(N−k)
             + [if N even: 0.5 × C(N, N/2) × p^(N/2) × (1−p)^(N/2)]
```

**Definition 2 — Correlated ensemble quality:**

```
Q(N, p, ρ) = p + (Q_ind(N, p) − p) × (1 − ρ)
```

where:
- `p ∈ (0.5, 1]`: per-agent accuracy (probability of correct output)
- `ρ ∈ [0, 1]`: mean pairwise error correlation (0 = independent, 1 = always err together)
- Boundary: N=1 → Q=p; ρ=1 → Q=p (no ensemble benefit when all agents err identically)

**Why this model applies:** H2AI runs multiple LLM adapters on the same task and takes a
consensus (verification-filtered merge). If each adapter is independently likely to produce
a correct output and their errors are not perfectly correlated, the majority outcome is
more likely correct than any single adapter. This is precisely the Condorcet setting.

**What this model does NOT claim:**
- It does not claim LLMs vote on a binary correct/incorrect decision. In practice,
  p and ρ are proxied from output similarity (CG_mean), not measured from a reference answer set.
- It does not claim outputs are independent. Adapters running the same base model at
  different temperatures have correlated outputs; ρ > 0 captures this.

**Validated by:** `scripts/validate_ensemble_theory.py` — Monte Carlo simulation confirms
the formula matches empirical voting outcomes at 100k trials per parameter set (Δ < 2%).

---

## 2. Parameter Estimation

### 2.1 Common Ground (CG)

**Definition 3:**

```
CG(i, j) = jaccard(K_i, K_j) × tau_alignment(τ_i, τ_j)
```

where:
- `K_i` = vocabulary set of adapter i's output tokens
- `τ_i` = creativity temperature of adapter i
- `tau_alignment(τ_i, τ_j) = exp(−3 × |τ_i − τ_j|)` ∈ (0, 1]

During calibration, all adapters run at the same `calibration_tau`, so `tau_alignment = 1.0`.
The factor is applied in code so the formula is correct when multi-τ calibration is introduced.

**CG_mean** is the mean of all pairwise CG values across calibration adapters.

**What CG measures:** Vocabulary overlap of outputs, not semantic agreement. High CG means
adapters used similar words; it does not guarantee they reached the same conclusion.

### 2.2 Accuracy and Correlation Proxies

When no reference eval set is available, H2AI derives p and ρ from CG_mean:

```
p_mean   = 0.5 + CG_mean / 2   ∈ [0.5, 1.0]
rho_mean = 1 − CG_mean          ∈ [0, 1]
```

**Rationale:** High CG_mean indicates adapters produce similar vocabulary, suggesting they
frequently agree (high p), but also that their errors are correlated (high ρ). At CG=1,
p=1 and ρ=0, giving Q=1. At CG→0, p=0.5 and ρ=1, giving Q=p=0.5.

**Limitation:** These are operational proxies, not measured accuracies. For production
deployments, run `scripts/baseline_eval.py` and set `baseline_accuracy_proxy` in config
to override the proxy with a measured value.

### 2.3 N_optimal

```
N_optimal = argmax_{N=1..9} [ (Q(N, p_mean, rho_mean) − p_mean) / N ]
```

This is the **marginal Condorcet gain per agent** above the single-agent baseline.
N=1 always scores 0 (no gain over itself); the formula finds the N where each additional
agent contributes the most incremental quality. The cap of 9 is a practical deployment limit.

---

## 3. Contention and Coordination

`CoherencyCoefficients` is produced by `CalibrationHarness` and used for topology provisioning.

```
alpha      — measured from calibration timing via Amdahl's law inverse.
             CalibrationHarness runs all M adapters concurrently on calibration prompts,
             measures wall-clock time T_parallel and per-adapter times T_i, then derives:

               speedup        = (ΣT_i) / T_parallel
               alpha_measured = ((M / speedup) − 1) / (M − 1)
               alpha          = clamp(alpha_measured, 0.05, 0.5)

             Falls back to config `alpha_contention` (default 0.12) when M < 2
             or timing is degenerate (speedup ≤ 1.0).

kappa_base = kappa_eff_factor × (2 − CG_mean)    [informational / telemetry only]
             Not used in N_max computation. Retained in the struct for API consumers
             that surface coordination cost estimates.

N_max      = floor(1 / alpha)    [Amdahl's law]
             Beyond 1/α agents, marginal throughput gain drops below 1% of single-agent
             baseline. Acts as a safety ceiling; Condorcet N_optimal is the primary target.
```

**Provenance of α:** Amdahl (1967). The formula `speedup = 1 / (α + (1−α)/M)` gives the
theoretical maximum speedup for M workers given serial fraction α. Inverting this to
solve for α from an observed speedup is standard practice in parallel performance analysis.

**What this measures:** Actual I/O and scheduling serialization during concurrent adapter
execution on the calibration host. It reflects the observed parallelism of the calibration
run, not a theoretical model of the task domain.

**What this does NOT measure:** Per-task contention under production load, adapter-specific
concurrency limits, or network saturation effects beyond the calibration prompts.

---

## 4. Attribution Model

**Implemented in:** `crates/h2ai-orchestrator/src/attribution.rs`

```
baseline_quality   = p_mean
topology_gain      = Q(N, p_mean, rho_mean) − p_mean     [Condorcet gain]
tao_multiplier     = tao_per_turn_factor ^ (turns − 1)
error_remaining    = (1 − Q(N, p_mean, rho_mean)) × verification_filter_ratio × tao_multiplier
total_quality      = 1 − error_remaining,  clamped to [p_mean, 1.0]
```

`topology_gain` is the marginal ensemble quality improvement predicted by Condorcet JT.
`tao_gain` and `verification_gain` are upper-bound estimates of per-phase contributions;
they are informational and do not partition `total_quality` additively.

---

## 5. J_eff — Context Adequacy Gate

**Implemented in:** `crates/h2ai-context/src/compiler.rs`, `crates/h2ai-context/src/similarity.rs`

### 5.1 Formula

```
j_positive    = semantic_jaccard(task_manifest, required_keywords, adapter)
contamination = |prohibited_terms ∩ tokenize(task_manifest)| / max(1, |tokenize(task_manifest)|)
J_eff         = j_positive × (1 − contamination)
```

where `required_keywords = corpus.vocabulary() ∪ manifest.explicit_constraints`.

### 5.2 Semantic Mode (adapter provided)

When a `similarity_adapter: Option<&dyn IComputeAdapter>` is passed to `compile()`,
`semantic_jaccard` dispatches a JSON-scoring prompt to the SLM adapter at near-zero
temperature (`SIMILARITY_TAU = 0.05`) and extracts a float score ∈ [0, 1]:

```
System: "You are a semantic-similarity oracle. Reply ONLY with JSON: {\"score\": <float 0-1>}"
Task:   "Score semantic similarity:\nA: {manifest}\nB: {required_keywords}"
```

The score is the true semantic overlap — synonyms, paraphrases, and domain-equivalent
terms all contribute, regardless of surface-level word choice.

**Fallback chain:** adapter error → token Jaccard; score out of [0, 1] → token Jaccard.

### 5.3 Token Mode (no adapter)

When `adapter` is `None`, `semantic_jaccard` falls back immediately to:

```
semantic_jaccard(a, b, None) = jaccard(tokenize(a), tokenize(b))
```

This is the deterministic, dependency-free mode used in unit tests and offline scenarios.

### 5.4 What J_eff Measures

**With semantic mode:**
- Vocabulary overlap _and_ semantic coverage — "payment throttling" scores high against
  "budget pacing" constraints because the SLM recognises the domain equivalence.
- Contamination detects prohibited terms from `ConstraintPredicate::NegativeKeyword` ADRs
  and penalises the score proportionally.

**With token mode (fallback):**
- Word-level vocabulary overlap only. Synonyms score 0. Use only when no SLM is available.

### 5.5 Known False Negatives (token mode only)

Token mode produces false negatives when the task description uses domain synonyms not
present in the constraint corpus vocabulary. Example: "payment throttling" vs "budget pacing"
gives token Jaccard ≈ 0.2 but semantic score ≈ 0.8. Always run with a `similarity_adapter`
in production to avoid false ContextUnderflowErrors on valid tasks.

### 5.6 Vocabulary Stuffing Resistance (semantic mode)

Token mode can be trivially gamed by appending constraint vocabulary to an off-domain task.
Semantic mode is resistant: the SLM scores the _dominant semantic content_ of the full text,
not keyword presence. An "adversarial" task that appends "budget pacing redis idempotency"
to an unrelated blockchain prompt scores ~0.14 semantically despite token Jaccard = 1.0.

**Gate:** J_eff < `j_eff_gate` (default 0.4) → `ContextUnderflowError`. This prevents
the Auditor from operating without adequate semantic coverage of the constraint space.

---

## 6. Semantic Cluster Coherence

**Implemented in:** `crates/h2ai-state/src/krum.rs`

Krum and Multi-Krum BFT selection require an honest cluster assumption: honest agent outputs
must cluster tightly in metric space so Krum can distinguish them from Byzantine outliers.
This assumption holds for _semantic_ distance but not necessarily for _lexical_ distance —
LLM paraphrases of the same solution may share few tokens while being semantically identical.

### 6.1 Mean Pairwise Distance

```
mean_pairwise_distance(proposals, adapter)
  = mean over all (i, j) pairs of (1 − semantic_jaccard(output_i, output_j, adapter))
```

All pairwise `semantic_jaccard` calls are dispatched concurrently via `join_all`.

**Token fallback:** When `adapter` is `None`, uses `1 − jaccard(tokenize(i), tokenize(j))`.

### 6.2 Cluster Coherence Guard

```
cluster_coherent(proposals, adapter) = mean_pairwise_distance(proposals, adapter) < MAX_CLUSTER_DIAMETER
```

where `MAX_CLUSTER_DIAMETER = 0.7` (constant in `krum.rs`).

**Effect:** Before applying Krum BFT selection, the merger checks whether the surviving proposals
form a coherent cluster. If `cluster_coherent` returns false, the Blanchard et al. geometric
assumption is violated and Krum's BFT guarantee does not hold. The merger falls back to
`ConsensusMedian` (majority-nearest-neighbour), which is robust to a divergent cluster
even though it is not Byzantine-fault-tolerant.

**Why semantic distance matters:** If honest agents produce lexically diverse paraphrases of
the same correct answer (high token distance, low semantic distance), token-based
`mean_pairwise_distance` would incorrectly classify a coherent cluster as incoherent,
triggering a needless ConsensusMedian fallback. Semantic distance avoids this.

---

## 7. Known Limitations and Future Work

| Limitation | Current mitigation | Future path |
|---|---|---|
| p and ρ proxied from CG_mean, not measured | `baseline_accuracy_proxy` config override | Per-task accuracy via reference eval sets |
| τ alignment always 1.0 during calibration | Documented — all adapters run same τ | Multi-τ calibration with role-specific prompts |
| N_optimal assumes uniform inference cost | T_synthesis = T_inference approximation | Measure actual synthesis latency |
| Condorcet assumes majority vote; H2AI uses merge | Merge + verification approximates majority | Direct accuracy measurement of merge outcome |
| semantic_jaccard latency adds to J_eff gate | SIMILARITY_TAU=0.05 keeps SLM calls deterministic and short | Cache per (manifest, required_kw) pair across tasks |
| Cluster coherence check is O(n²) pairwise calls | join_all parallelises all calls; n ≤ 9 in practice | Batch embedding endpoint for ≥ 10 proposals |

---

## 8. Simulation Evidence

```bash
python3 scripts/validate_ensemble_theory.py
python3 scripts/validate_ensemble_theory.py --plot  # generates charts in scripts/
```

The simulation verifies:

**Condorcet / Ensemble:**
1. Formula boundary conditions — N=1 → Q=p; ρ=1 → Q=p
2. Monotonicity — Q non-decreasing in N for p > 0.5, ρ < 1
3. Monte Carlo match — empirical voting at 100k trials matches formula within 2%
4. Proxy sensibility — derived p and ρ produce valid n_optimal values

**Semantic J_eff:**
5. Synonym gap — domain-equivalent phrases (different words) score near 0 with token Jaccard
   but above gate with semantic mode; confirms false-negative fix
6. Vocabulary stuffing resistance — off-domain text + appended constraint keywords scores
   token Jaccard = 1.0 but semantic score ≈ 0.1–0.2; confirms false-positive mitigation
7. None-adapter fallback equivalence — `semantic_jaccard(..., None)` produces identical
   result to direct token Jaccard

**Cluster coherence:**
8. Semantic paraphrase cluster — proposals that are lexically distant but semantically
   equivalent score low mean semantic distance (< MAX_CLUSTER_DIAMETER), confirming
   `cluster_coherent` returns `true` and Krum proceeds correctly
9. Genuinely diverse cluster — proposals from unrelated domains score high mean semantic
   distance (> MAX_CLUSTER_DIAMETER), confirming `cluster_coherent` returns `false`
   and ConsensusMedian fallback is triggered
