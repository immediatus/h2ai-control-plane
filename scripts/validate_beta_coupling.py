#!/usr/bin/env python3
"""
Validate β_eff coupling formula as implemented in h2ai-state/calibration.rs:

    β_eff = β₀ / max(CG_mean, 0.05)

The collapse floor max(CG, 0.05) prevents singularity as CG → 0
and triggers the N_max=1 guard when CG < 0.10.

Key invariants validated:
  1. CG=1.0  → β_eff = β₀             (perfect alignment, no extra cost)
  2. CG=0.4  → β_eff = 2.5 × β₀      (typical AI-agent tier)
  3. CG=0.05 → β_eff = 20 × β₀       (floor active)
  4. CG=0.02 → same as CG=0.05        (floor clamps singularity)
  5. CG < 0.10 → N_max=1              (collapse guard)

See docs/architecture/math-apparatus.md — Definition 4, Proposition 1.
"""

import numpy as np
import sys

# ── helpers ──────────────────────────────────────────────────────────────────

CG_FLOOR = 0.05
CG_COLLAPSE_THRESHOLD = 0.10  # below this, N_max is forced to 1


def beta_eff(beta0: float, cg: float) -> float:
    """β_eff = β₀ / max(CG, CG_FLOOR)"""
    return beta0 / max(cg, CG_FLOOR)


def n_max(alpha: float, beta0: float, cg: float) -> int:
    """N_max = round(√((1−α) / β_eff)); returns 1 when CG < collapse threshold."""
    if cg < CG_COLLAPSE_THRESHOLD:
        return 1
    be = beta_eff(beta0, cg)
    return max(1, round(np.sqrt((1.0 - alpha) / be)))


failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")
    if not ok:
        failures.append(name)


# ── 1. Key invariants ─────────────────────────────────────────────────────────

beta0 = 0.039
alpha = 0.15

print("=" * 72)
print("1.  Key Invariants — β_eff = β₀ / max(CG, 0.05)")
print("=" * 72)

be_cg1 = beta_eff(beta0, 1.0)
check(
    "CG=1.0 → β_eff = β₀ (no extra coherency cost at perfect alignment)",
    abs(be_cg1 - beta0) < 1e-10,
    f"β_eff={be_cg1:.6f}, β₀={beta0}"
)

be_cg04 = beta_eff(beta0, 0.4)
check(
    "CG=0.4 → β_eff = 2.5 × β₀ (typical AI-agent tier)",
    abs(be_cg04 / beta0 - 2.5) < 0.01,
    f"β_eff={be_cg04:.6f} = {be_cg04/beta0:.2f} × β₀"
)

be_floor = beta_eff(beta0, 0.05)
check(
    "CG=0.05 → β_eff = 20 × β₀ (floor active, not a singularity)",
    abs(be_floor / beta0 - 20.0) < 0.01,
    f"β_eff={be_floor:.6f} = {be_floor/beta0:.2f} × β₀"
)

be_below_floor = beta_eff(beta0, 0.02)
check(
    "CG=0.02 → β_eff = β_eff(0.05) (floor clamps further divergence)",
    abs(be_below_floor - be_floor) < 1e-10,
    f"β_eff(0.02)={be_below_floor:.6f} == β_eff(0.05)={be_floor:.6f}"
)

# ── 2. Three-tier N_max ────────────────────────────────────────────────────────

TIERS = [
    # (label, alpha, beta_base, cg_mean, n_max_expected)
    ("AI agents",    0.15, 0.01,   0.40, 6),
    ("Human teams",  0.10, 0.005,  0.60, 10),
    ("CPU cores",    0.02, 0.0003, 1.00, 57),
]

print()
print("=" * 72)
print("2.  Three-Tier N_max  (β_eff = β₀ / max(CG, 0.05))")
print("=" * 72)
print(f"{'Tier':<15} {'CG':>6} {'β_eff':>10} {'N_max':>7} {'Expected':>9}")
print("-" * 55)
for name, a, b0, cg, n_exp in TIERS:
    be = beta_eff(b0, cg)
    nm = n_max(a, b0, cg)
    marker = "✓" if abs(nm - n_exp) <= 1 else "✗"
    print(f"{name:<15} {cg:>6.2f} {be:>10.5f} {nm:>7} {n_exp:>9}  {marker}")
    check(
        f"N_max ≈ {n_exp} [{name}]",
        abs(nm - n_exp) <= 1,
        f"got N_max={nm}"
    )

# ── 3. Collapse guard ─────────────────────────────────────────────────────────

print()
print("=" * 72)
print("3.  CG Collapse Guard — N_max forced to 1 when CG < 0.10")
print("=" * 72)
for cg in [0.09, 0.05, 0.02, 0.01]:
    nm = n_max(alpha, beta0, cg)
    check(
        f"CG={cg:.2f} → N_max=1 (ZeroCoordinationQualityEvent emitted)",
        nm == 1,
        f"got N_max={nm}"
    )

# ── 4. Monotonicity above collapse floor ──────────────────────────────────────

print()
print("=" * 72)
print("4.  Monotonicity — higher CG → lower β_eff (above collapse floor)")
print("=" * 72)
cg_range = np.linspace(0.10, 1.0, 20)
be_values = [beta_eff(beta0, cg) for cg in cg_range]
check(
    "β_eff strictly decreasing as CG increases from 0.10 to 1.0",
    all(be_values[i] > be_values[i + 1] for i in range(len(be_values) - 1)),
    f"min β_eff={min(be_values):.5f} at CG=1.0, max β_eff={max(be_values):.5f} at CG=0.10"
)

# ── 5. Sweep CG from 0.05 to 0.99 ─────────────────────────────────────────────

print()
print("=" * 72)
print("5.  Full sweep — AI agents tier (α=0.15, β₀=0.039)")
print("=" * 72)
print(f"{'CG':>6} {'β_eff':>10} {'N_max':>7}")
print("-" * 30)
for cg in np.linspace(0.05, 1.0, 20):
    be = beta_eff(beta0, cg)
    nm = n_max(alpha, beta0, cg)
    print(f"{cg:6.2f} {be:>10.5f} {nm:>7}")

# ── Results ───────────────────────────────────────────────────────────────────

print()
print("=" * 72)
n_fail = len(failures)
print(f"Failures: {n_fail}")
if failures:
    for f in failures:
        print(f"  • {f}")
    sys.exit(1)
else:
    print("PASS — all checks passed")
    sys.exit(0)
