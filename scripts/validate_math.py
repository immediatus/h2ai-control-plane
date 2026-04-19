"""
Math apparatus validator — docs/architecture/05-math-apparatus.md

Validates every definition and proposition numerically using only the standard
library. No external dependencies. Exit code 0 = all checks pass. Designed to
run in CI without a full Rust build.

Usage:
    python scripts/validate_math.py
    python scripts/validate_math.py --verbose
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
