"""
Math apparatus validator — docs/architecture/05-math-apparatus.md

Validates every definition and proposition numerically using only the standard
library. No external dependencies. Exit code 0 = all checks pass. Designed to
run in CI without a full Rust build.

WHEN TO RUN
-----------
Run this script any time you change a formula, constant, or threshold in
docs/architecture/05-math-apparatus.md. A failing check is the signal that the
doc and the implementation have diverged.

    python scripts/validate_math.py            # normal output (PASS/FAIL per check)
    python scripts/validate_math.py --verbose  # include detail lines on passing checks too

Exit code 0 = all checks pass. Non-zero = at least one failure (printed at end).

CROSS-REFERENCE MAP  (doc section → functions/constants in this file)
----------------------------------------------------------------------
§ Definition 1  (α)           — parameter in usl_throughput(); values in CALIBRATION_TABLE
§ Definition 2  (USL formula) — usl_throughput(), usl_throughput_extended()
§ Definition 3  (CG)          — jaccard(), tau_alignment(), common_ground()
§ Definition 4  (κ_eff)       — CALIBRATION_TABLE κ_eff column checks
§ Definition 5  (Extended USL)— usl_throughput_extended(); algebraic equivalence in calibration
§ Definition 6  (Edge Count)  — edge_count_flat(), edge_count_tree()
§ Definition 7  (RW Graph)    — c_i parameter in byzantine_loss(), merge_strategy()
§ Definition 8  (Byz. Loss)   — byzantine_loss()
§ Definition 9  (Pareto axes) — entropy() implements H(τ) (D axis)
§ Definition 10 (J_eff)       — j_eff(), jaccard(); J_EFF_GATE = 0.4
§ Proposition 1 (N_max)       — analytical_n_max(), numerical_n_max(), retrograde checks
§ Proposition 2 (Conway)      — coordination_threshold()
§ Proposition 3 (Condorcet)   — majority_vote_accuracy(), correlated_ensemble_accuracy()
§ Proposition 4 (Entropy)     — entropy()
§ Proposition 5 (Safety)      — edge_count_flat/tree(), merge_strategy(); BFT_THRESHOLD = 0.85
§ §3 Calibration table        — CALIBRATION_TABLE (must match doc table exactly)
§ §4 Safety constraints       — J_EFF_GATE = 0.4, BFT_THRESHOLD = 0.85

CONSTANTS THAT MUST STAY IN SYNC WITH THE DOC
----------------------------------------------
    J_EFF_GATE    = 0.4   → docs/architecture/05-math-apparatus.md §4 (J_eff gate row)
    BFT_THRESHOLD = 0.85  → docs/architecture/05-math-apparatus.md §Proposition 5 safety constraint
    CALIBRATION_TABLE     → §3 Calibration and §Proposition 1 calibrated ceilings table
"""

import math
import sys
import itertools
from typing import Callable

VERBOSE = "--verbose" in sys.argv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail and (VERBOSE or not condition):
        msg += f"\n         {detail}"
    print(msg)
    if not condition:
        _failures.append(name)


def approx_eq(a: float, b: float, tol: float = 0.01) -> bool:
    """Relative tolerance comparison."""
    if b == 0:
        return abs(a) < tol
    return abs(a - b) / abs(b) < tol


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f" {title}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Definition 2 — USL throughput formula
# ---------------------------------------------------------------------------

def usl_throughput(N: float, alpha: float, kappa: float) -> float:
    """X(N) = N / (1 + α(N-1) + κ·N(N-1))"""
    return N / (1 + alpha * (N - 1) + kappa * N * (N - 1))


def usl_throughput_extended(N: float, alpha: float, kappa_base: float, cg_mean: float) -> float:
    kappa_eff = kappa_base / cg_mean
    return usl_throughput(N, alpha, kappa_eff)


# § Definition 2 — docs/architecture/05-math-apparatus.md
section("Definition 2 — USL Throughput Formula")

# Single agent baseline: X(1) must equal 1 for all α, κ
for alpha, kappa in [(0.02, 0.0003), (0.10, 0.005), (0.15, 0.025)]:
    result = usl_throughput(1, alpha, kappa)
    check(
        f"X(1) = 1.0 for α={alpha}, κ={kappa}",
        approx_eq(result, 1.0, tol=1e-9),
        f"got X(1) = {result}"
    )

# Throughput must be positive for all valid N
check(
    "X(N) > 0 for N in [1..20] (hardware layer)",
    all(usl_throughput(n, 0.02, 0.0003) > 0 for n in range(1, 21))
)

# ---------------------------------------------------------------------------
# Definition 4 — Effective Coherency
# ---------------------------------------------------------------------------

# § Definition 4 — docs/architecture/05-math-apparatus.md
section("Definition 4 — Effective Coherency  κ_eff = κ_base / CG_mean")

CALIBRATION_TABLE = [
    # (layer, alpha, kappa_base, cg_mean, kappa_eff_expected, n_max_expected)
    ("CPU cores",    0.02, 0.0003, 1.0, 0.0003, 57),
    ("Human teams",  0.10, 0.005,  0.6, 0.0083, 10),
    ("AI agents",    0.15, 0.01,   0.4, 0.025,   6),
]

for layer, alpha, kappa_base, cg_mean, kappa_eff_exp, _ in CALIBRATION_TABLE:
    kappa_eff_calc = kappa_base / cg_mean
    check(
        f"κ_eff = κ_base / CG_mean  [{layer}]",
        approx_eq(kappa_eff_calc, kappa_eff_exp, tol=0.05),
        f"κ_base={kappa_base}, CG_mean={cg_mean} → κ_eff={kappa_eff_calc:.5f} (expected ≈{kappa_eff_exp})"
    )

# Higher CG → lower κ_eff (agents share more context, pay less coordination cost)
kappa_eff_low_cg  = 0.01 / 0.4   # AI agents
kappa_eff_high_cg = 0.01 / 1.0   # same κ_base, perfect CG
check(
    "Higher CG_mean → lower κ_eff (invariant)",
    kappa_eff_high_cg < kappa_eff_low_cg,
    f"CG=1.0 → κ_eff={kappa_eff_high_cg:.4f}; CG=0.4 → κ_eff={kappa_eff_low_cg:.4f}"
)

# ---------------------------------------------------------------------------
# Proposition 1 — Scalability Ceiling  N_max = sqrt((1-α) / κ_eff)
# ---------------------------------------------------------------------------

# § Proposition 1 — docs/architecture/05-math-apparatus.md
section("Proposition 1 — Scalability Ceiling")


def analytical_n_max(alpha: float, kappa: float) -> float:
    """N_max = sqrt((1-α) / κ)"""
    return math.sqrt((1 - alpha) / kappa)


def numerical_n_max(alpha: float, kappa: float, n_range: int = 200) -> float:
    """Find N that maximises X(N) numerically."""
    best_n, best_x = 1, 0.0
    for n in range(1, n_range + 1):
        x = usl_throughput(n, alpha, kappa)
        if x > best_x:
            best_x = x
            best_n = n
    return float(best_n)


for layer, alpha, kappa_base, cg_mean, kappa_eff, n_max_exp in CALIBRATION_TABLE:
    n_analytical = analytical_n_max(alpha, kappa_eff)
    n_numerical  = numerical_n_max(alpha, kappa_eff)

    check(
        f"Analytical N_max ≈ {n_max_exp}  [{layer}]",
        approx_eq(n_analytical, n_max_exp, tol=0.15),
        f"formula → {n_analytical:.1f}, expected ≈{n_max_exp}"
    )
    check(
        f"Analytical N_max matches numerical peak  [{layer}]",
        abs(n_analytical - n_numerical) <= 2.0,
        f"analytical={n_analytical:.1f}, numerical peak at N={n_numerical}"
    )

# Retrograde: throughput at N=2*N_max must be less than at N=N_max
for layer, alpha, kappa_base, cg_mean, kappa_eff, n_max_exp in CALIBRATION_TABLE:
    x_at_peak    = usl_throughput(n_max_exp, alpha, kappa_eff)
    x_past_peak  = usl_throughput(n_max_exp * 2, alpha, kappa_eff)
    check(
        f"Retrograde: X(2·N_max) < X(N_max)  [{layer}]",
        x_past_peak < x_at_peak,
        f"X(N_max={n_max_exp})={x_at_peak:.3f}, X({n_max_exp*2})={x_past_peak:.3f}"
    )

# Derivative is zero at N_max (proposition proof sanity check)
# dX/dN ≈ 0 at N_max; check sign change: positive just before, negative just after
for layer, alpha, kappa_base, cg_mean, kappa_eff, n_max_exp in CALIBRATION_TABLE:
    n_peak = analytical_n_max(alpha, kappa_eff)
    delta = 0.5
    slope_before = usl_throughput(n_peak - delta, alpha, kappa_eff) - usl_throughput(n_peak - delta - 0.1, alpha, kappa_eff)
    slope_after  = usl_throughput(n_peak + delta + 0.1, alpha, kappa_eff) - usl_throughput(n_peak + delta, alpha, kappa_eff)
    check(
        f"dX/dN changes sign at N_max (maximum confirmed)  [{layer}]",
        slope_before > 0 and slope_after < 0,
        f"slope before N_max: {slope_before:.4f} (should be +), after: {slope_after:.4f} (should be -)"
    )

# ---------------------------------------------------------------------------
# Definition 3 — Common Ground
# ---------------------------------------------------------------------------

# § Definition 3 — docs/architecture/05-math-apparatus.md
section("Definition 3 — Common Ground  CG(i,j) = J(K_i, K_j) × alignment(τ_i, τ_j)")


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def tau_alignment(tau_i: float, tau_j: float) -> float:
    """Monotonically decreasing in |τ_i - τ_j|. Simple linear proxy."""
    return max(0.0, 1.0 - abs(tau_i - tau_j))


def common_ground(K_i: set, K_j: set, tau_i: float, tau_j: float) -> float:
    return jaccard(K_i, K_j) * tau_alignment(tau_i, tau_j)


# Identical agents → CG = 1.0
K = {"auth", "redis", "kafka", "jwt"}
check(
    "CG(i, i) = 1.0 (identical agent)",
    approx_eq(common_ground(K, K, 0.5, 0.5), 1.0),
)

# Disjoint knowledge → CG = 0.0
check(
    "CG = 0.0 when knowledge bases are disjoint",
    approx_eq(common_ground({"A", "B"}, {"C", "D"}, 0.5, 0.5), 0.0),
)

# Same knowledge but opposite temperatures → CG reduces
cg_same_tau  = common_ground(K, K, 0.1, 0.1)
cg_diff_tau  = common_ground(K, K, 0.1, 0.9)
check(
    "CG decreases as |τ_i - τ_j| increases (same K)",
    cg_diff_tau < cg_same_tau,
    f"CG(same τ)={cg_same_tau:.3f}, CG(diff τ)={cg_diff_tau:.3f}"
)

# CG is symmetric
K_i = {"auth", "redis", "grpc"}
K_j = {"auth", "kafka", "rest"}
check(
    "CG(i, j) = CG(j, i) (symmetry)",
    approx_eq(
        common_ground(K_i, K_j, 0.3, 0.7),
        common_ground(K_j, K_i, 0.7, 0.3),
    )
)

# CG_mean < 1 → κ_eff > κ_base (agents pay extra coordination cost)
cg_mean_example = 0.4
kappa_base_example = 0.01
kappa_eff_example = kappa_base_example / cg_mean_example
check(
    "CG_mean < 1.0 → κ_eff > κ_base",
    kappa_eff_example > kappa_base_example,
    f"κ_base={kappa_base_example}, CG_mean={cg_mean_example} → κ_eff={kappa_eff_example}"
)

# ---------------------------------------------------------------------------
# Definition 10 — Dark Knowledge Gap (J_eff)
# ---------------------------------------------------------------------------

# § Definition 10 — docs/architecture/05-math-apparatus.md  |  J_EFF_GATE must equal §4 table
section("Definition 10 — Dark Knowledge Gap  J_eff = J(K_prompt, K_task_required)")

J_EFF_GATE = 0.4


def j_eff(k_prompt: set, k_required: set) -> float:
    return jaccard(k_prompt, k_required)


# Full coverage → J_eff = 1.0
k_task = {"budget-idempotency", "redis-atomic", "kafka-billing", "session-compliance"}
check(
    "J_eff = 1.0 when prompt covers all required knowledge",
    approx_eq(j_eff(k_task, k_task), 1.0),
)

# Empty prompt → J_eff = 0.0 → gate triggers
check(
    "J_eff = 0.0 for empty prompt → ContextUnderflowError",
    j_eff(set(), k_task) < J_EFF_GATE,
    f"J_eff={j_eff(set(), k_task)}, gate={J_EFF_GATE}"
)

# Partial prompt: 2 of 4 concepts covered
k_partial = {"budget-idempotency", "redis-atomic"}
j = j_eff(k_partial, k_task)
check(
    "J_eff = 0.5 for 2-of-4 coverage → above gate",
    j >= J_EFF_GATE,
    f"J_eff={j:.3f}, gate={J_EFF_GATE}"
)

# ADR adds constraints → J_eff increases monotonically as corpus grows
k_no_adr   = {"describe the task"}
k_one_adr  = k_no_adr | {"budget-idempotency", "redis-atomic"}
k_two_adrs = k_one_adr | {"kafka-billing", "session-compliance"}
j_no_adr   = j_eff(k_no_adr,   k_task)
j_one_adr  = j_eff(k_one_adr,  k_task)
j_two_adrs = j_eff(k_two_adrs, k_task)
check(
    "J_eff increases monotonically as ADR corpus grows",
    j_no_adr < j_one_adr < j_two_adrs,
    f"no ADR={j_no_adr:.3f}, 1 ADR={j_one_adr:.3f}, 2 ADRs={j_two_adrs:.3f}"
)

# ---------------------------------------------------------------------------
# Definition 8 — Byzantine Expected Loss
# ---------------------------------------------------------------------------

# § Definition 8 — docs/architecture/05-math-apparatus.md
section("Definition 8 — Byzantine Expected Loss  L_i = c_i × P(hallucination) × propagation")


def byzantine_loss(c_i: float, p_hallucination: float, propagation: int) -> float:
    return c_i * p_hallucination * propagation


N_agents = 5

# Flat topology: propagation = N - 1
loss_flat = byzantine_loss(c_i=0.8, p_hallucination=0.2, propagation=N_agents - 1)

# Tree with branching factor k=2: propagation = k
loss_tree = byzantine_loss(c_i=0.8, p_hallucination=0.2, propagation=2)

# Review gate quarantines the fault: propagation = 1
loss_gate = byzantine_loss(c_i=0.8, p_hallucination=0.2, propagation=1)

check(
    "L(flat) > L(tree) > L(review-gate) for same c_i and p_hallucination",
    loss_flat > loss_tree > loss_gate,
    f"flat={loss_flat:.3f}, tree={loss_tree:.3f}, gate={loss_gate:.3f}"
)

# Total expected loss: flat mesh vs hierarchical tree (N=5 agents)
# Flat: each agent propagates to N-1 = 4 peers
total_loss_flat = sum(
    byzantine_loss(0.5, 0.15, N_agents - 1) for _ in range(N_agents)
)
# Tree (k=2): each leaf propagates to k=2 peers; coordinator absorbs the rest
total_loss_tree = sum(
    byzantine_loss(0.5, 0.15, 2) for _ in range(N_agents)
)
check(
    "Total expected loss: flat > tree (N=5 agents)",
    total_loss_flat > total_loss_tree,
    f"total flat={total_loss_flat:.3f}, total tree={total_loss_tree:.3f}"
)

# ---------------------------------------------------------------------------
# Proposition 2 — Epistemic Conway Constraint
# ---------------------------------------------------------------------------

# § Proposition 2 — docs/architecture/05-math-apparatus.md
section("Proposition 2 — Epistemic Conway Constraint")


def coordination_threshold(cg_values: list[float]) -> float:
    """θ_coord = min(CG_mean - σ_CG, 0.3)"""
    n = len(cg_values)
    mean = sum(cg_values) / n
    variance = sum((x - mean) ** 2 for x in cg_values) / n
    sigma = math.sqrt(variance)
    return min(mean - sigma, 0.3)


# Well-aligned team: θ_coord ≤ CG values
cg_high = [0.65, 0.70, 0.60, 0.68]
theta = coordination_threshold(cg_high)
check(
    "θ_coord ≤ 0.3 for well-aligned team",
    theta <= 0.3,
    f"θ_coord={theta:.4f}"
)
check(
    "All CG values ≥ θ_coord (coordination viable)",
    all(cg >= theta for cg in cg_high),
    f"min CG={min(cg_high):.2f}, θ_coord={theta:.4f}"
)

# Misaligned team: one pair below θ_coord signals misplaced boundary
cg_mixed = [0.65, 0.70, 0.18, 0.68]  # one pair with very low CG
theta_mixed = coordination_threshold(cg_mixed)
check(
    "CG < θ_coord detected for misaligned pair",
    any(cg < theta_mixed for cg in cg_mixed),
    f"min CG={min(cg_mixed):.2f}, θ_coord={theta_mixed:.4f} — boundary is misplaced"
)

# ---------------------------------------------------------------------------
# Proposition 3 — Multiplication Condition
# ---------------------------------------------------------------------------

# § Proposition 3 — docs/architecture/05-math-apparatus.md
section("Proposition 3 — Multiplication Condition (Generalised Condorcet)")


def majority_vote_accuracy(individual_accuracy: float, n_agents: int) -> float:
    """
    Probability that majority vote is correct when each agent has independent
    accuracy p > 0.5. Uses the exact binomial sum.
    """
    majority = n_agents // 2 + 1
    total = 0.0
    p = individual_accuracy
    for k in range(majority, n_agents + 1):
        # C(n,k) * p^k * (1-p)^(n-k)
        binom = math.comb(n_agents, k)
        total += binom * (p ** k) * ((1 - p) ** (n_agents - k))
    return total


# Condition 1 check: p > 0.5 required for benefit
p_good  = 0.7
p_bad   = 0.4  # below 0.5 — majority vote hurts

acc_good_3 = majority_vote_accuracy(p_good, 3)
acc_bad_3  = majority_vote_accuracy(p_bad,  3)

check(
    "Majority vote with p=0.7 beats individual accuracy",
    acc_good_3 > p_good,
    f"individual={p_good}, 3-agent majority={acc_good_3:.4f}"
)
check(
    "Majority vote with p=0.4 is WORSE than individual (Cond 1 violated)",
    acc_bad_3 < p_bad,
    f"individual={p_bad}, 3-agent majority={acc_bad_3:.4f}"
)

# Condorcet benefit grows monotonically with N (given p > 0.5)
accs = [majority_vote_accuracy(p_good, n) for n in [1, 3, 5, 7]]
check(
    "Majority vote accuracy increases with N when p > 0.5",
    all(accs[i] < accs[i+1] for i in range(len(accs) - 1)),
    f"N=1,3,5,7 → {[f'{a:.4f}' for a in accs]}"
)

# Condition 2: correlated errors cancel the benefit
# Simulate: if two agents share errors perfectly (ρ=1), adding the second does nothing
def correlated_ensemble_accuracy(p: float, n: int, rho: float) -> float:
    """
    Approximate: error decorrelation factor scales benefit by (1 - rho^(n-1)).
    At rho=1 all agents fail together; at rho=0 errors are independent.
    """
    if n == 1:
        return p
    independent_acc = majority_vote_accuracy(p, n)
    # Interpolate: correlated ensemble is between individual and independent
    return p + (independent_acc - p) * (1 - rho ** (n - 1))

acc_uncorrelated = correlated_ensemble_accuracy(p_good, 5, rho=0.0)
acc_correlated   = correlated_ensemble_accuracy(p_good, 5, rho=0.95)
check(
    "Highly correlated errors (ρ=0.95) reduce ensemble benefit",
    acc_correlated < acc_uncorrelated,
    f"ρ=0.0 → {acc_uncorrelated:.4f}, ρ=0.95 → {acc_correlated:.4f}"
)

# ---------------------------------------------------------------------------
# Proposition 4 — Merge Semantics and Epistemic Entropy
# ---------------------------------------------------------------------------

# § Proposition 4 — docs/architecture/05-math-apparatus.md
section("Proposition 4 — Merge Semantics and Epistemic Entropy")


def entropy(probs: list[float]) -> float:
    """Shannon entropy H(τ)."""
    return -sum(p * math.log2(p) for p in probs if p > 0)


# Agent temperature distribution before merge
tau_distribution = [0.1, 0.3, 0.5, 0.7, 0.9]
n_tau = len(tau_distribution)
uniform_probs = [1 / n_tau] * n_tau

H_before = entropy(uniform_probs)

# Consensus: all agents align to mode (single τ dominates)
consensus_probs = [0.0, 0.0, 1.0, 0.0, 0.0]  # majority at τ=0.5
H_consensus = entropy(consensus_probs)

# CRDT: all contributions preserved (uniform distribution maintained)
H_crdt = entropy(uniform_probs)  # unchanged

check(
    "Consensus merge collapses H(τ) → 0",
    H_consensus < 0.01,
    f"H_before={H_before:.3f}, H_consensus={H_consensus:.3f}"
)
check(
    "CRDT merge preserves H(τ)",
    approx_eq(H_crdt, H_before),
    f"H_before={H_before:.3f}, H_crdt={H_crdt:.3f}"
)
check(
    "H(CRDT) > H(consensus) — CRDT preserves epistemic diversity",
    H_crdt > H_consensus,
    f"H_crdt={H_crdt:.3f}, H_consensus={H_consensus:.3f}"
)

# ---------------------------------------------------------------------------
# Proposition 5 — CRDT-Merge Hierarchy Dominance
# ---------------------------------------------------------------------------

# § Proposition 5 — docs/architecture/05-math-apparatus.md  |  BFT_THRESHOLD must equal §Prop 5
section("Proposition 5 — CRDT-Merge Hierarchy Dominance + Safety Constraint")


def edge_count_flat(n: int) -> int:
    return n * (n - 1) // 2


def edge_count_tree(n: int) -> int:
    return n - 1


# Edge count: tree < flat for N > 2
for n in [3, 5, 6, 10]:
    check(
        f"E_tree < E_flat for N={n}  (hierarchy reduces coordination cost)",
        edge_count_tree(n) < edge_count_flat(n),
        f"E_tree={edge_count_tree(n)}, E_flat={edge_count_flat(n)}"
    )

# Throughput: tree topology stays below its own N_max longer
# Flat mesh: N_max ≈ 6 for AI agents (κ_eff = 0.025)
# Tree (coordinator absorbs coherency): effective κ reduced by branching factor k
k = 3  # branching factor
alpha_ai = 0.15
kappa_flat = 0.025
kappa_tree = kappa_flat / k  # coordinator absorbs most coherency cost

n_max_flat = analytical_n_max(alpha_ai, kappa_flat)
n_max_tree = analytical_n_max(alpha_ai, kappa_tree)

check(
    "Tree topology has higher N_max than flat mesh (same α, reduced κ_eff)",
    n_max_tree > n_max_flat,
    f"N_max(flat)={n_max_flat:.1f}, N_max(tree, k={k})={n_max_tree:.1f}"
)

# Safety constraint: BFT required when max(c_i) > 0.85
BFT_THRESHOLD = 0.85

def merge_strategy(role_error_costs: list[float]) -> str:
    return "BftConsensus" if max(role_error_costs) > BFT_THRESHOLD else "CrdtSemilattice"

check(
    "CrdtSemilattice selected when all c_i ≤ 0.85",
    merge_strategy([0.3, 0.5, 0.7, 0.85]) == "CrdtSemilattice",
)
check(
    "BftConsensus selected when any c_i > 0.85",
    merge_strategy([0.3, 0.5, 0.9]) == "BftConsensus",
)
check(
    "BFT threshold is exactly 0.85 (boundary value)",
    merge_strategy([0.851]) == "BftConsensus" and merge_strategy([0.85]) == "CrdtSemilattice",
    "c_i=0.851 → BFT, c_i=0.85 → CRDT"
)

# ---------------------------------------------------------------------------
# Calibration table — full cross-check
# ---------------------------------------------------------------------------

# § §3 Calibration + §Proposition 1 — docs/architecture/05-math-apparatus.md
# CALIBRATION_TABLE values must exactly match both tables in the doc
section("Calibration Reference Table — Full Cross-Check")

for layer, alpha, kappa_base, cg_mean, kappa_eff_exp, n_max_exp in CALIBRATION_TABLE:
    kappa_eff_calc = kappa_base / cg_mean
    n_max_calc = analytical_n_max(alpha, kappa_eff_calc)

    check(
        f"[{layer}] κ_eff = {kappa_eff_exp}",
        approx_eq(kappa_eff_calc, kappa_eff_exp, tol=0.06),
        f"κ_base/CG_mean = {kappa_base}/{cg_mean} = {kappa_eff_calc:.5f}"
    )
    check(
        f"[{layer}] N_max ≈ {n_max_exp}",
        approx_eq(n_max_calc, n_max_exp, tol=0.20),
        f"sqrt((1-{alpha})/{kappa_eff_calc:.5f}) = {n_max_calc:.1f}"
    )

# Extended USL N_max formula with CG_mean factored in
for layer, alpha, kappa_base, cg_mean, kappa_eff_exp, n_max_exp in CALIBRATION_TABLE:
    n_max_extended = math.sqrt((1 - alpha) * cg_mean / kappa_base)
    n_max_simple   = math.sqrt((1 - alpha) / (kappa_base / cg_mean))
    check(
        f"[{layer}] Extended USL N_max form is algebraically equivalent",
        approx_eq(n_max_extended, n_max_simple, tol=1e-6),
        f"extended={n_max_extended:.3f}, simple={n_max_simple:.3f}"
    )

# ---------------------------------------------------------------------------
# §6–§7 Harness Physics Extensions (Definitions 11–13, Propositions 6–8)
# ---------------------------------------------------------------------------

# § Definition 11 — TAO Error Reduction  c_i_effective(t) = c_i × 0.60^(t-1)
# TAO_DECAY = 0.60 must match simulate_usl.py TAO_DECAY constant
section("Definition 11 — TAO Error Reduction  c_i × 0.60^(t-1)")

TAO_DECAY = 0.60


def tao_error_reduction(c_i: float, turns: int) -> float:
    return c_i * (TAO_DECAY ** (turns - 1))


check("t=1 → c_i unchanged",
      approx_eq(tao_error_reduction(0.5, 1), 0.5),
      f"got {tao_error_reduction(0.5, 1)}")
check("t=2 → 60% of c_i",
      approx_eq(tao_error_reduction(0.5, 2), 0.30),
      f"got {tao_error_reduction(0.5, 2)}")
check("t=3 → 36% of c_i",
      approx_eq(tao_error_reduction(0.5, 3), 0.18),
      f"got {tao_error_reduction(0.5, 3)}")
check("monotone decrease across turns",
      tao_error_reduction(0.5, 3) < tao_error_reduction(0.5, 2) < tao_error_reduction(0.5, 1))
check("Shell agent (c_i=0.9) drops below BFT threshold at t=2",
      tao_error_reduction(0.9, 2) < 0.85,
      f"c_i_eff={tao_error_reduction(0.9, 2):.3f} at t=2")
check("Executor (c_i=0.5) already below BFT threshold at t=1",
      tao_error_reduction(0.5, 1) < 0.85,
      f"c_i={tao_error_reduction(0.5, 1):.3f}")
check("Shell agent exact crossover value t=2 ≈ 0.540",
      approx_eq(tao_error_reduction(0.9, 2), 0.540, tol=0.01),
      f"got {tao_error_reduction(0.9, 2):.3f}")

# § Definition 12 — Verification Filter Gain
# verification_gain(c_i, filter_ratio) = c_i × (1 - filter_ratio)
section("Definition 12 — Verification Filter Gain  c_i × (1 − filter_ratio)")


def verification_gain(c_i: float, filter_ratio: float) -> float:
    return c_i * (1.0 - filter_ratio)


check("zero gain at filter_ratio=1.0 (no filtering)",
      approx_eq(verification_gain(0.5, 1.0), 0.0))
check("maximum gain at filter_ratio=0.0 (all filtered)",
      approx_eq(verification_gain(0.5, 0.0), 0.5))
check("Executor +25pp at 50% filter",
      approx_eq(verification_gain(0.5, 0.5), 0.25),
      f"got {verification_gain(0.5, 0.5):.3f}")
check("Shell agent +45pp at 50% filter",
      approx_eq(verification_gain(0.9, 0.5), 0.45),
      f"got {verification_gain(0.9, 0.5):.3f}")
check("monotone: lower filter_ratio → higher gain",
      verification_gain(0.5, 0.3) > verification_gain(0.5, 0.7))

# § Definition 13 — Harness Attribution Decomposition
# Q_total = Q_baseline + G_topology + G_tao + G_verify  (clamped to 1.0)
section("Definition 13 — Harness Attribution  Q_total = baseline + topology + tao + verify")

ALPHA_AI = 0.15
KAPPA_AI = 0.025


def harness_q_total(c_i: float, n: int, alpha: float, kappa_e: float,
                    filter_ratio: float, tao_turns: float) -> float:
    baseline = 1.0 - c_i
    n = max(n, 1)
    usl_n = n / (1.0 + alpha * (n - 1) + kappa_e * n * (n - 1))
    usl_n = max(usl_n, 1.0)
    g_topo = max(c_i * (1.0 - 1.0 / usl_n), 0.0)
    tao_c = c_i * (TAO_DECAY ** (tao_turns - 1))
    g_tao = max((1.0 - tao_c) - baseline, 0.0)
    g_verify = c_i * (1.0 - filter_ratio)
    return min(baseline + g_topo + g_tao + g_verify, 1.0)


check("Q_total ≥ baseline (1-c_i) — harness never reduces quality",
      harness_q_total(0.5, 1, ALPHA_AI, KAPPA_AI, 1.0, 1) >= (1.0 - 0.5),
      f"got {harness_q_total(0.5, 1, ALPHA_AI, KAPPA_AI, 1.0, 1):.3f}")
check("ensemble improves over single agent (N=4 > N=1)",
      harness_q_total(0.5, 4, ALPHA_AI, KAPPA_AI, 1.0, 1) >=
      harness_q_total(0.5, 1, ALPHA_AI, KAPPA_AI, 1.0, 1))
check("TAO improves over no TAO",
      harness_q_total(0.5, 4, ALPHA_AI, KAPPA_AI, 1.0, 3) >
      harness_q_total(0.5, 4, ALPHA_AI, KAPPA_AI, 1.0, 1))
check("full harness reaches quality ceiling (≤ 1.0)",
      harness_q_total(0.5, 4, ALPHA_AI, KAPPA_AI, 0.5, 3) <= 1.0)
check("Q_total always non-negative",
      harness_q_total(0.0, 1, 0.0, 0.001, 1.0, 1) >= 0.0)

# § Proposition 6 — Parallel Verification Speedup
# T(N, P) = ceil(N/P) × T_eval
section("Proposition 6 — Parallel Verification  T(N,P) = ceil(N/P) × T_eval")

check("P=N → single T_eval (fully parallel)",
      math.ceil(6 / 6) == 1)
check("P=1 → N × T_eval (fully sequential)",
      math.ceil(6 / 1) == 6)
check("P=3 → 2 × T_eval",
      math.ceil(6 / 3) == 2)
check("P=N/2 → 3× speedup vs P=1  (T(6,3)=2 < T(6,1)=6)",
      math.ceil(6 / 3) < math.ceil(6 / 1),
      f"T(6,3)={math.ceil(6/3)}, T(6,1)={math.ceil(6/1)}")

# § Proposition 7 — TAO Convergence turns
# t* = ceil(log(ε/c_i) / log(0.6)) to reach error floor ε
section("Proposition 7 — TAO Convergence  t* = ceil(log(ε/c_i) / log(TAO_DECAY))")


def tao_convergence_turns(c_i: float, target_eps: float) -> int:
    if c_i <= target_eps:
        return 1
    return math.ceil(math.log(target_eps / c_i) / math.log(TAO_DECAY))


# Brute-force verification
def brute_convergence(c_i: float, target_eps: float) -> int:
    for t in range(1, 30):
        if tao_error_reduction(c_i, t) <= target_eps:
            return t
    return 30


check("formula matches brute force for c_i=0.5, eps=0.1",
      abs(tao_convergence_turns(0.5, 0.1) - brute_convergence(0.5, 0.1)) <= 1,
      f"formula={tao_convergence_turns(0.5, 0.1)}, brute={brute_convergence(0.5, 0.1)}")
check("formula matches brute force for c_i=0.9, eps=0.1",
      abs(tao_convergence_turns(0.9, 0.1) - brute_convergence(0.9, 0.1)) <= 1,
      f"formula={tao_convergence_turns(0.9, 0.1)}, brute={brute_convergence(0.9, 0.1)}")
check("Shell agent crosses BFT threshold in ≤2 turns",
      brute_convergence(0.9, BFT_THRESHOLD) <= 2,
      f"turns={brute_convergence(0.9, BFT_THRESHOLD)}")

# § Proposition 8 — Attribution Monotonicity
# Q_total is monotone in N (≤N_max), TAO turns, and (1-filter_ratio)
section("Proposition 8 — Attribution Monotonicity  ∂Q/∂N≥0, ∂Q/∂t≥0, ∂Q/∂(1-f)≥0")

n_max_ai = int(analytical_n_max(ALPHA_AI, KAPPA_AI))
q_by_n = [harness_q_total(0.5, n, ALPHA_AI, KAPPA_AI, 1.0, 1) for n in range(1, n_max_ai + 1)]
check("Q_total monotone in N (1..N_max)",
      all(q_by_n[i] <= q_by_n[i+1] for i in range(len(q_by_n)-1)),
      f"N=1..{n_max_ai} values: {[f'{v:.4f}' for v in q_by_n]}")

q_by_tao = [harness_q_total(0.5, 4, ALPHA_AI, KAPPA_AI, 1.0, float(t)) for t in range(1, 5)]
check("Q_total monotone in TAO turns (1..4)",
      all(q_by_tao[i] <= q_by_tao[i+1] for i in range(len(q_by_tao)-1)),
      f"t=1..4 values: {[f'{v:.4f}' for v in q_by_tao]}")

q_by_fr = [harness_q_total(0.5, 4, ALPHA_AI, KAPPA_AI, fr, 1.0)
           for fr in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]]
check("Q_total monotone in verification strictness (filter_ratio 1.0→0.0)",
      all(q_by_fr[i] <= q_by_fr[i+1] for i in range(len(q_by_fr)-1)),
      f"fr=1.0..0.0 values: {[f'{v:.4f}' for v in q_by_fr]}")

check("TAO gain magnitude: t=1→4 matches simulation finding (+21.88pp ±1pp)",
      abs((q_by_tao[-1] - q_by_tao[0]) * 100 - 21.88) < 1.0,
      f"Δ={( q_by_tao[-1] - q_by_tao[0])*100:.2f}pp")
check("Marginal topology gain N=4→N_max smaller than TAO gain t=1→2",
      (q_by_n[-1] - q_by_n[-2]) < (q_by_tao[1] - q_by_tao[0]),
      f"Δ(N+1)={(q_by_n[-1]-q_by_n[-2])*100:.2f}pp < Δ(t+1)={(q_by_tao[1]-q_by_tao[0])*100:.2f}pp")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

section("Results")

total = sum(1 for _ in [  # count total checks by re-running is hard; count failures instead
    None
])

n_fail = len(_failures)
print(f"\n  Failures: {n_fail}")
if _failures:
    print("\n  Failed checks:")
    for f in _failures:
        print(f"    • {f}")
    print()
    sys.exit(1)
else:
    print("  All checks passed.\n")
    sys.exit(0)
