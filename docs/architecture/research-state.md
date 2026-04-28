# H2AI Research State

Consolidated theory, implemented math, research gap analysis, and validation evidence
for the H2AI Control Plane. This is the single authoritative document linking mathematical
claims to implementation and empirical proof.

---

## 1. Project Thesis

**One equation governs CPU caches, human teams, and AI agent swarms** — for the same
structural reason.

Whenever N nodes must maintain mutual consistency, pairwise synchronization cost grows as
N(N−1). A CPU cache coherency protocol exchanges cache lines between every pair of cores.
A human engineering team (Brook's Law, 1975) opens N−1 new communication channels for
each new member. An AI agent ensemble must check every pair of partial answers for
semantic contradiction before merge. The phenomenon is identical at all three scales;
only the physical substrate changes.

The Universal Scalability Law (Gunther 1993) describes this precisely:

```
X(N) = N / (1 + α(N−1) + β·N(N−1))

N_max = √((1−α) / β_eff)
```

Two opposing forces act on ensemble quality. The **quality force** (Condorcet Jury Theorem):
each additional independent agent reduces the probability that a wrong answer reaches the
output — quality converges to 1 as N grows. The **coordination force** (USL): reconciling
N agents' outputs requires O(N²) pairwise consistency checks — beyond N_max, this cost
exceeds the Condorcet gain and adding agents actively degrades results.

Their intersection is the computable optimal N. CJT quantifies the quality gain force;
USL quantifies the coordination cost force.

The β parameter (coherency coefficient) is modulated by **Common Ground** — the semantic
overlap between agents' outputs. When agents share high CG (compatible partial solutions),
reconciliation is cheap. When they have split, each pair is expensive to reconcile. The
coupling is:

```
β_eff = β₀ × (1 − CG_mean)
```

This formula is proportional and bounded everywhere (floor 1e-6). The previous inverse
form β₀/CG_mean diverged at CG→0 and was replaced in commit 0093c4d.

---

## 2. Implemented Math Apparatus

### 2.1 Condorcet Jury Theorem — Ensemble Quality

**Source:** Condorcet (1785), Nitzan & Paroush (1982), Ladha (1992).  
**Implemented in:** `h2ai-types/src/physics.rs` — `condorcet_quality()`  
**Validated by:** `scripts/validate_ensemble_theory.py` — Monte Carlo matches formula within 2% at 100k trials.

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
- `p ∈ (0.5, 1]`: per-agent accuracy
- `ρ ∈ [0, 1]`: mean pairwise error correlation (0 = independent, 1 = always err together)
- Boundary: N=1 → Q=p; ρ=1 → Q=p (no ensemble benefit when all agents err identically)

**What this model does NOT claim:**
- It does not claim LLMs vote on a binary correct/incorrect decision.
- It does not claim outputs are independent. Adapters from the same model family at different temperatures have correlated errors; ρ > 0 captures this.
- The proxy `p_mean = 0.5 + CG_mean/2` is a heuristic, not a measurement. Use `baseline_eval.py` for high-stakes deployments.

### 2.2 Common Ground (CG)

**Implemented in:** `h2ai-types/src/physics.rs` — pairwise CG computation in calibration  
**Validated by:** `scripts/validate_ensemble_theory.py` — proxy sensibility confirmed.

**Definition 3:**

```
CG(i, j) = jaccard(K_i, K_j) × tau_alignment(τ_i, τ_j)
```

where:
- `K_i` = vocabulary set of adapter i's output tokens
- `tau_alignment(τ_i, τ_j) = exp(−3 × |τ_i − τ_j|)` ∈ (0, 1]

During calibration, all adapters run at the same `calibration_tau`, so `tau_alignment = 1.0`.

**What CG measures:** Vocabulary overlap of outputs, not semantic agreement. High CG means
adapters used similar words; it does not guarantee they reached the same conclusion.

### 2.3 Accuracy and Correlation Proxies

**Implemented in:** `h2ai-types/src/physics.rs` — `EnsembleCalibration`

```
p_mean   = 0.5 + CG_mean / 2   ∈ [0.5, 1.0]
rho_mean = 1 − CG_mean          ∈ [0, 1]
```

**Limitation:** These are operational proxies, not measured accuracies. Run
`scripts/baseline_eval.py` and set `baseline_accuracy_proxy` to override with a measured
value for production deployments.

### 2.4 N_optimal — Marginal Condorcet Gain

**Implemented in:** `h2ai-types/src/physics.rs` — `EnsembleCalibration::n_optimal()`

```
N_optimal = argmax_{N=1..9} [ (Q(N, p_mean, rho_mean) − p_mean) / N ]
```

The marginal Condorcet gain per agent above the single-agent baseline. N=1 always scores 0;
the formula finds the N where each additional agent contributes the most incremental quality.
The cap of 9 is a practical deployment limit.

### 2.5 N_it_optimal — Information-Theoretic Ensemble Size

**Implemented in:** `h2ai-types/src/physics.rs` — `n_it_optimal()` and `EnsembleCalibration::n_it_optimal()`  
**Validated by:** `scripts/validate_information_theory.py` — I_marginal exponential decay confirmed.

```
N_it_optimal = ⌈1 + ln(0.5) / ln(1 − ρ)⌉    clamped to [1, 9]
```

Derived from `I_marginal(N) = H(X) × (1−ρ)^(N−1)`: the marginal information gain drops
below half of per-adapter entropy when `(1−ρ)^(N−1) < 0.5`.

| ρ | N_it_optimal | interpretation |
|---|---|---|
| 0.0 | 1 | independent agents — single is sufficient |
| 0.3 | 3 | mild correlation |
| 0.5 | 2 | moderate correlation |
| 1.0 | 9 | fully correlated — cap applies |

Matches Condorcet N_optimal within ±1 for ρ ∈ [0.3, 0.95].

### 2.6 USL Throughput and Two-Phase Calibration

**Implemented in:** `h2ai-types/src/physics.rs` — `usl_fit()` and `CoherencyCoefficients`  
**Validated by:** `scripts/validate_ensemble_theory.py` — two-phase parameter recovery error < 0.01 (α) and < 0.002 (β₀).

```
X(N) = N / (1 + α(N−1) + β·N(N−1))
```

**Two-phase calibration:**

```
Phase A: run 2 adapters → T₁ (per-adapter time) and T₂ (wall clock)
Phase B: run M adapters → T_M (wall clock) and adapter outputs for CG_mean

Linearization: z(N) = N·T_parallel(N)/T₁ − 1 = α(N−1) + β₀·N(N−1)

Analytical solution from z₂ and z_M:
  β₀ = (z_M − z₂·(M−1)) / ((M−1)(M−2))    [only valid when M ≥ 3]
  α  = z₂ − 2·β₀

Clamped: α → [0.05, 0.5], β₀ → [1e-6, 0.1]
Falls back to config alpha_contention and beta_base_default when M < 3
```

**Provenance:** Gunther (1993). The two-phase fit replaces the earlier single-phase Amdahl
inverse which derived only α.

### 2.7 β_eff — USL+CG Coupling

**Implemented in:** `h2ai-types/src/physics.rs` — `CoherencyCoefficients::beta_eff()`  
**Validated by:** `scripts/validate_beta_coupling.py` — proportional form bounded everywhere; inverse form diverges at CG→0.

```
β_eff = β₀ × (1 − CG_mean)    [Definition 6 — proportional, bounded]
floor: max(β_eff, 1e-6)

At CG_mean → 0: β_eff = β₀     (maximum; agents have split)
At CG_mean = 1: β_eff → 0      (agents fully aligned; only α limits N_max)
```

The previous inverse form `β₀/CG_mean` diverged at CG→0 and was replaced.

### 2.8 N_max — Scalability Ceiling

**Implemented in:** `h2ai-types/src/physics.rs` — `CoherencyCoefficients::n_max()`

```
N_max = round(√((1 − α) / β_eff))    [USL Proposition 1]
```

Derived by setting dX/dN = 0. Beyond N_max the USL throughput curve enters retrograde
(throughput decreasing with more agents).

**Three-tier calibration table (proportional formula):**

| Domain | α | β₀ | CG_mean | β_eff | N_max |
|---|---|---|---|---|---|
| AI agents | 0.15 | 0.039 | 0.4 | 0.0234 | ≈ 6 |
| Human teams | 0.10 | 0.0225 | 0.6 | 0.009 | ≈ 10 |
| CPU cores | 0.02 | 0.0003 | 1.0 | ≈ 0 (→ α only) | large |

**Validated by:** `scripts/simulate_usl.py` — item 13: N_max Proposition 1 gives N_max≈6 (AI), ≈10 (Human), ≈57 (CPU)

### 2.9 EigenCalibration — N_eff Participation Ratio

**Source:** Portfolio theory (Choueifaty & Coignard 2008).  
**Implemented in:** `h2ai-types/src/physics.rs` — `EigenCalibration::from_cg_matrix()`  
**Validated by:** `scripts/validate_eigenvalue_calibration.py` — formula validated against uniform and heterogeneous correlation matrices; scalar ρ overstates N_eff by 55% on heterogeneous structures.

```
Σ ∈ ℝ^(N×N)   pairwise CG similarity matrix (Σ_ij = CG(adapter_i, adapter_j))
λ_1 ≥ ... ≥ λ_N = eigenvalues of Σ

N_eff  = (Σ λᵢ)² / Σ λᵢ²        [participation ratio]
H_norm = (−Σ (λᵢ/Σλ) × log(λᵢ/Σλ)) / log(N)    [normalized diversity ∈ [0,1]]
ρ_eff  = 1 − N_eff/N             [effective correlation from matrix]
```

N_eff is strictly more informative than scalar ρ_mean. Example: 5 adapters with 2
independent + 3 in a tight cluster → N_eff ≈ 2.5, but scalar CG_mean proxy gives
N_eff_scalar ≈ 3.9 (over-estimate by 55%).

**Adapter pruning rule:** Add adapter N+1 only if N_eff increases by ≥ 0.05.

### 2.10 Temporal Decay for CG Calibration

**Implemented in:** `h2ai-types/src/physics.rs` — `CoherencyCoefficients::beta_eff_temporal()`

```
w(t_i) = exp(-(now_secs − t_i) / CG_HALFLIFE_SECS)

CG_eff = Σ(CG_i × w(t_i)) / Σ(w(t_i))

beta_eff_temporal = max(beta_base × (1 − CG_eff), 1e-6)
```

where `CG_HALFLIFE_SECS = 604_800` (7 days). As all samples age past ~35 half-lives,
`CG_eff → 0` and `beta_eff_temporal → beta_base` — creating natural pressure to re-calibrate.

**Fallback asymmetry:**
- Mismatched/empty timestamps → `beta_eff()` (neutral unweighted result)
- Weight exhaustion (all samples ancient) → `beta_base` (conservative ceiling)

### 2.11 Attribution Model

**Implemented in:** `h2ai-orchestrator/src/attribution.rs`

```
baseline_quality   = p_mean
topology_gain      = Q(N, p_mean, rho_mean) − p_mean     [Condorcet gain]
tao_multiplier     = tao_per_turn_factor ^ (turns − 1)
error_remaining    = (1 − Q(N, p_mean, rho_mean)) × verification_filter_ratio × tao_multiplier
total_quality      = 1 − error_remaining,  clamped to [p_mean, 1.0]
```

**Validated by:** no dedicated script — formula correctness verified by unit tests in `h2ai-orchestrator/src/attribution.rs`

### 2.12 J_eff — Context Adequacy Gate

**Implemented in:** `h2ai-context/src/compiler.rs`, `h2ai-context/src/similarity.rs`  
**Validated by:** `scripts/validate_ensemble_theory.py` — synonym gap and vocabulary stuffing resistance confirmed.

```
j_positive    = semantic_jaccard(task_manifest, required_keywords, adapter)
contamination = |prohibited_terms ∩ tokenize(task_manifest)| / max(1, |tokenize(task_manifest)|)
J_eff         = j_positive × (1 − contamination)
```

Gate: J_eff < `j_eff_gate` (default 0.4) → `ContextUnderflowError`.

### 2.13 Semantic Cluster Coherence (Krum Guard)

**Implemented in:** `h2ai-state/src/krum.rs`  
**Validated by:** `scripts/validate_bft_methods.py` — honest selection rate with semantic vs token distance.

```
mean_pairwise_distance = mean over all (i,j) pairs of (1 − semantic_jaccard(output_i, output_j))
cluster_coherent       = mean_pairwise_distance < MAX_CLUSTER_DIAMETER (0.7)
```

If `cluster_coherent` returns false, the Blanchard et al. geometric assumption is violated
and Krum falls back to `ConsensusMedian` (Weiszfeld geometric median, breakdown point 50%).

### 2.14 Talagrand Rank Histogram Diagnostic

**Implemented in:** `h2ai-orchestrator/src/diagnostics.rs` — `TalagrandDiagnostic`  
**Inspired by:** Leutbecher & Palmer (2008, ECMWF).

```
For run t with N proposals, sort scores descending: s₁ ≥ ... ≥ s_N.
Rank r_t = position of runner-up in score ordering.
Histogram H[r] = count{t : r_t = r}.
Uniformity: χ² = Σ_r (H[r] − T/N)² / (T/N)    calibrated iff χ² < 3.84
```

Flat → well-calibrated; U-shape → over-confident (expand τ spread);
Λ-shape → under-dispersed (try diverse model families). No ground-truth labels needed.

**Validated by:** `scripts/validate_eigenvalue_calibration.py` Section 4 (adapter pruning) and `scripts/validate_information_theory.py` Section 3 (Slepian-Wolf efficiency)

### 2.15 RRF Fusion

**Implemented in:** `h2ai-context/src/fusion.rs`  
**Source:** Cormack, Clarke & Buettcher (2009), SIGIR.

```
rrf_score(d) = Σ_i  1 / (k + rank_i(d))
```

where `k = 60` (standard constant). `hybrid_search()` fuses a token Jaccard stream and an
embedding cosine stream via RRF — preventing either from dominating when they disagree.

---

## 3. Research Gaps and Fix Status

### Gap P1 — USL Domain Transfer (partially addressed)

**Problem:** USL was derived for shared-state distributed systems (CPU caches, database pools).
Applying it to LLM ensembles requires a domain transfer argument. The original single-phase
calibration (Amdahl inverse) measured only α, not β₀ directly. With only one adapter timing
data point, the fit is under-constrained and sensitive to timing noise.

**Status:** The two-phase fit (Phase A: N=2, Phase B: N=M) now solves analytically for both
α and β₀ from two timing measurements. This is an improvement over single-phase estimation.
Multi-point least-squares fitting (M > 3 adapters with varied timing) remains future work and
would improve β₀ precision on noisy measurements. The calibration falls back to config defaults
when M < 3.

### Gap P2 — CJT N_optimal Not Wired to Planner (in plan, Task 3)

**Problem:** `EigenCalibration` computes N_eff and N_it_optimal, but the planner selects
ensemble size from config defaults rather than from these computed values. The physics
computation happens but has no downstream effect on topology provisioning.

**Status:** Wiring EigenCalibration output into the planner's ensemble size decision is
planned as Task 3. Until wired, N_optimal is advisory only; N_max from USL is the only
physics-derived constraint currently enforced in provisioning.

### Gap P3 — BFT in Semantic Space — ConsensusMedian Has No Metric Guarantee (in plan, Task 4)

**Problem:** `ConsensusMedian` computes the Fréchet median on token-level representations.
Token Jaccard distance is not a proper metric in embedding space — paraphrases of identical
content score high distance, breaking the clustering assumption that Krum and median require.
The breakdown point guarantee (Vardi & Zhang 2000) assumes the metric is Euclidean or at
least a proper distance function.

**Status:** Replacing the token-based median with Weiszfeld geometric median on embeddings
(Pillutla et al. 2019) is planned as Task 4. Weiszfeld provides a provable 50% breakdown
point in proper metric spaces.

### Gap P4 — β_eff Formula (FIXED — commit 0093c4d)

**Problem:** The original formula `β_eff = β₀ / CG_mean` diverges as CG_mean → 0.
At low Common Ground (when agents have split most severely — exactly when the formula
matters most), it produces unbounded N_max values that cause numerical instability and
nonsensical topology decisions.

**Status:** Fixed. The proportional formula `β_eff = β₀ × (1 − CG_mean)` is implemented
with floor 1e-6. At CG→0, β_eff = β₀ (maximum finite coherency cost). At CG=1,
β_eff ≈ 0 (agents fully aligned; α alone limits N_max). Validated by
`scripts/validate_beta_coupling.py`.

### Gap P5 — Calibration Bypass When 1 Adapter Passes (in plan, Task 1)

**Problem:** When the calibration harness receives only 1 adapter instead of the expected M ≥ 3,
the USL fit returns None and the system falls back to config defaults. This means a common
deployment mistake (single adapter configured) silently bypasses the physics-derived N_max
and uses uncalibrated defaults for all task topologies.

**Status:** Enforcing a minimum of 3 adapters during calibration (returning a hard error
rather than a silent fallback when M < 3) is planned as Task 1. The config fields
`calibration_adapter_count` and `calibration_tau_spread` will be added to make the
multi-adapter calibration intent explicit.

### Gap P6 — CRDT Monotonicity — No Generation Counter (in plan, Task 2)

**Problem:** The CRDT semilattice merge does not carry a generation counter (vector clock).
Without a monotonicity guarantee, concurrent writes from two adapters with the same
logical timestamp can produce non-deterministic merge outcomes that depend on arrival order
rather than the CRDT join semilattice property.

**Status:** Adding version vectors to CRDT state is planned as Task 2. Until added, the
merge is deterministic only for proposals with strictly ordered timestamps.

---

## 4. Innovation Synthesis

Three independent research domains — portfolio theory, BFT federated learning, and
information theory — all converge on the same conclusions when applied to LLM ensemble
design.

### Cross-Domain Convergence

**1. Scalar ρ is wrong — the full CG similarity matrix is needed.**

Portfolio theory (Choueifaty & Coignard 2008) proves that the participation ratio
N_eff = (Σλ)²/Σλ² from the eigenvalue decomposition of the pairwise correlation matrix
is strictly more informative than any scalar summary of that matrix. Applied to LLM
ensembles: a group of 5 adapters where 2 are independent and 3 form a tight cluster has
N_eff ≈ 2.5 independent ideas, but scalar CG_mean gives N_eff_scalar ≈ 3.9 — a 55%
over-estimate that causes the planner to deploy more agents than are informationally useful.

BFT federated learning reaches the same conclusion differently: Krum's geometric argument
requires that honest proposals cluster in a proper metric space. Scalar ρ says nothing
about the geometry of that cluster.

**2. Lexical distance is wrong everywhere — embedding space is the correct metric.**

Krum (Blanchard et al. 2017) and Weiszfeld (Vardi & Zhang 2000) both require a proper
metric. Token Jaccard fails this requirement: two paraphrases of identical answers share
few tokens but are semantically identical. The cluster coherence check in `krum.rs`
addresses this locally; the ConsensusMedian fallback still uses token-level representations
and is the planned fix in Task 4.

**3. Static calibration is a design smell — online learning complements static physics.**

The two-phase USL fit runs at deployment time and produces fixed α and β₀. These are
correct for the deployment conditions at calibration time. As models update, task domains
shift, and adapter pool composition changes, the calibration becomes stale. The Ebbinghaus
temporal decay (`beta_eff_temporal`) provides a partial remedy by down-weighting old CG
samples. True online adaptation — updating α and β₀ from observed NATS event timing spans
— remains future work.

### Key Innovations and Their Sources

**Eigenvalue N_optimal from portfolio theory**

Choueifaty & Coignard (2008) introduced the participation ratio as a measure of portfolio
diversification. Applied here: the participation ratio of the CG pairwise matrix measures
how many genuinely independent perspectives the adapter pool contains, independent of pool
size. This gives a principled stopping rule for adapter pruning (add adapter N+1 only if
N_eff increases by ≥ 0.05) and a direct proxy for ρ_eff that accounts for heterogeneous
correlation structure.

**Weiszfeld geometric median on embeddings**

Pillutla et al. (2019, arXiv:1912.13445) applied Weiszfeld's iterative algorithm for the
geometric median (Fréchet median in Euclidean space) to federated learning aggregation.
In that context, the geometric median is Byzantine-robust because it is the minimiser of
sum-of-distances, which Byzantine outliers can shift by at most f/(n−f) for f Byzantine
nodes. Vardi & Zhang (2000) proved that the Fréchet median has a 50% breakdown point —
it remains valid as long as honest nodes outnumber Byzantine ones. Applied to H2AI:
replacing token-level ConsensusMedian with Weiszfeld on embeddings gives a formally
guaranteed BFT aggregation method.

**Information-theoretic N_it_optimal**

The marginal information gain of the N-th agent is `I_marginal(N) = H(X) × (1−ρ)^(N−1)`.
This is an exponential decay in N. The natural stopping point is where this drops below
half the per-adapter entropy: `N_it_optimal = ⌈1 + ln(0.5)/ln(1−ρ)⌉`. This is derivable
from the Slepian-Wolf theorem on correlated sources and converges to the same N as the
Condorcet marginal gain formula within ±1 for ρ ∈ [0.3, 0.95].

**Talagrand rank histogram calibration diagnostic from weather forecasting**

Leutbecher & Palmer (2008) introduced the rank histogram to verify that ensemble weather
forecasts are statistically consistent — that the observed outcome is equally likely to
appear anywhere in the ranked ensemble. A flat histogram is the signature of a well-calibrated
ensemble; U-shape indicates over-confidence; Λ-shape indicates under-dispersion. The
diagnostic requires no ground-truth labels: it is derived purely from internal consistency
of the ensemble's score distribution. Applied to H2AI, it provides an operational signal
for when to adjust τ spread or model diversity.

---

## 5. Validation Evidence

Each mathematical claim in the implementation is formally verified by a script.

| Formula / Claim | Script | What it proves |
|---|---|---|
| β_eff = β₀×(1−CG) bounded everywhere; inverse form diverges | `validate_beta_coupling.py` | Proportional bounded at CG→0; inverse→∞ |
| Weiszfeld breakdown point 50%; Token Krum fails on paraphrases | `validate_bft_methods.py` | Honest selection rate vs Byzantine fraction f=1,2 |
| N_eff=(Σλ)²/Σλ² detects hidden redundancy scalar ρ misses | `validate_eigenvalue_calibration.py` | N_eff vs ρ_mean on heterogeneous correlation matrix |
| I_marginal=H(X)×(1−ρ)^(N-1); N_it_optimal within ±1 of Condorcet | `validate_information_theory.py` | Decay curve, Slepian-Wolf efficiency |
| CJT formula within 2% of Monte Carlo at 100k trials | `validate_ensemble_theory.py` | Monte Carlo vs formula; J_eff gate; proxy proxies |
| CJT over-predicts 5–15pp at ρ≥0.6; conformal set correct by construction | `validate_conformal_vs_cjt.py` | CJT vs conformal coverage at LLM-typical ρ |
| USL curves, Pareto matrix, attribution decomposition | `simulate_usl.py` | Visual shape of all equations (7 PNGs to scripts/output/) |

---

## 6. Script Catalog

### Production Tool

**`baseline_eval.py`** — Measures real per-adapter accuracy (p) and correlation (ρ) from
live adapters against `eval_questions.jsonl`. Run before high-stakes deployments to
override the CG_mean proxy with empirical values via the `baseline_accuracy_proxy` config
field. Output: per-adapter p and pairwise ρ matrix printed to stdout.

When to run: before deploying to a new task domain, when accuracy requirements are
contractual, or when the proxy-based N_optimal seems wrong for the observed task quality.

### Research and Validation Tools (not required for deployment)

These scripts are the formal proofs for specific mathematical claims. Run them after any
change to calibration constants or physics formulas to verify no formula regression.

**`validate_beta_coupling.py`**  
Validates the β_eff coupling formula. Shows the singularity of the inverse form
(β₀/CG→∞ at CG→0) and confirms the proportional form (β₀×(1−CG)) stays bounded.
Plots N_max vs CG_mean for both formulas.  
Expected output: proportional form produces finite N_max for all CG ∈ (0,1];
inverse form spikes to cap at CG < 0.1.  
When to run: after any change to `CoherencyCoefficients::beta_eff()` or the β₀ calibration.

**`validate_bft_methods.py`**  
Compares three BFT aggregation methods: Token Jaccard Krum, Embedding Krum, and Weiszfeld
geometric median. Uses synthetic embeddings to simulate honest/Byzantine proposals at
f=1,2 Byzantine fraction.  
Expected output: Token Krum honest selection rate ≈ 0.9% (cluster guard fires 100% with
90% paraphrase rate); Embedding Krum and Weiszfeld achieve 100% honest selection rate.  
When to run: after any change to `krum.rs` or the ConsensusMedian implementation.

**`validate_eigenvalue_calibration.py`**  
Validates the participation ratio N_eff = (Σλ)²/Σλ². Tests against uniform correlation
matrices (analytically solvable), heterogeneous matrices (2 independent + 3 clustered
adapters), and the adapter pruning stopping rule.  
Expected output: For heterogeneous structure, N_eff ≈ 2.5 vs scalar proxy N_eff_scalar
≈ 3.9. Pruning rule retains N=2 at ρ=0.9 (1.4 effective ideas from 9 adapters).  
When to run: after any change to `EigenCalibration::from_cg_matrix()`.

**`validate_information_theory.py`**  
Validates the information-theoretic N_it_optimal formula and I_marginal decay curve.
Also verifies Slepian-Wolf efficiency η = (1 + (N-1)(1-ρ)) / N.  
Expected output: N_it_optimal within ±1 of Condorcet N_optimal for ρ ∈ [0.3, 0.95];
Slepian-Wolf η drops below 0.5 at N=3 for ρ=0.7 (each additional adapter >50% redundant).  
When to run: after any change to `n_it_optimal()` or the ρ proxy formula.

**`validate_ensemble_theory.py`**  
Validates the CJT formula Q(N,p,ρ) against Monte Carlo simulation at 100k trials per
parameter set, and tests the J_eff gate and proxy sensibility.  
Expected output: formula matches Monte Carlo within 2%; boundary conditions N=1→Q=p and
ρ=1→Q=p hold; J_eff gate fires correctly at threshold 0.4.  
When to run: after any change to `condorcet_quality()` or the J_eff gate.

**`validate_conformal_vs_cjt.py`**  
Compares CJT quality prediction against conformal prediction set coverage at ρ ≥ 0.6
(LLM-typical same-family correlation).  
Expected output: CJT over-predicts quality 5–15 percentage points at ρ ≥ 0.6; conformal
set size > 1 correctly signals no consensus and triggers TAO retry.  
When to run: when evaluating ensemble quality claims at high ρ values, or when considering
replacing CJT with conformal coverage bounds.

**`validate_math.py`**  
Numerically validates every definition and proposition in `docs/architecture/math-apparatus.md`
using only the Python standard library — covering USL throughput, κ_eff, N_max, Common Ground,
J_eff, Byzantine loss, Condorcet majority vote, CRDT merge entropy, BFT hierarchy, calibration
table cross-check, and TAO/verification/attribution propositions.  
Expected output: all checks PASS (exit code 0); any failure indicates a formula divergence
between the doc and the implementation constants.  
When to run: after any change to a formula, constant, or threshold in
`docs/architecture/math-apparatus.md` or the corresponding physics implementation files.

**`simulate_usl.py`**  
Generates 7 visualisation plots to `scripts/output/`: USL curves for all three calibrated
layers, CG_mean effect on N_max, Pareto matrix heatmap, J_eff distribution, TAO error
reduction curves, attribution decomposition, and Q_total sensitivity plots.  
Expected output: 7 PNG files in `scripts/output/`. Correct shapes: USL peaks at N_max per
tier; proportional β_eff never diverges; attribution stacks sum correctly.  
When to run: when exploring theory, verifying visual shapes after parameter changes, or
for documentation diagrams.

---

## 7. References

- Gunther, N. J. (1993). A simple capacity model of massively parallel transaction systems. *CMG Conference Proceedings*. — Universal Scalability Law.
- Condorcet, M. J. A. N. (1785). *Essai sur l'application de l'analyse à la probabilité des décisions rendues à la pluralité des voix*. — Original Jury Theorem.
- Nitzan, S. & Paroush, J. (1982). Optimal decision rules in uncertain dichotomous choice situations. *International Economic Review*, 23(2), 289–297. — CJT restated formally.
- Ladha, K. K. (1992). The Condorcet Jury Theorem, free speech, and correlated votes. *American Journal of Political Science*, 36(3), 617–634. — CJT with correlated votes.
- Choueifaty, Y. & Coignard, Y. (2008). Toward maximum diversification. *Journal of Portfolio Management*, 35(1), 40–51. — Portfolio theory, participation ratio N_eff.
- Pillutla, V., Kakade, S. M., & Harchaoui, Z. (2019). Robust aggregation for federated learning. arXiv:1912.13445. — Weiszfeld geometric median for Byzantine-robust aggregation.
- Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). Machine learning with adversaries: Byzantine tolerant gradient descent. *NeurIPS*. — Krum BFT aggregation.
- Leutbecher, M. & Palmer, T. N. (2008). Ensemble forecasting. *Journal of Computational Physics*, 227(7), 3515–3539. (ECMWF) — Talagrand rank histogram calibration diagnostic.
- Lefort, T., et al. (2024). Empirical testing of the Condorcet Jury Theorem on large language models. arXiv:2409.00094. — CJT empirically tested on LLMs; ρ-dependent over-prediction at high correlation.
- Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. *SIGIR*. — RRF fusion.
- Vardi, Y. & Zhang, C.-H. (2000). The multivariate L1-median and associated data depth. *PNAS*, 97(4), 1423–1426. — Fréchet median breakdown point 50%.
- Brooks, F. P. (1975). *The Mythical Man-Month: Essays on Software Engineering*. Addison-Wesley. — Brook's Law: N(N−1)/2 communication channels.
