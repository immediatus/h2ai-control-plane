"""
H2AI visualization suite — docs/architecture/math.md

Six plots, saved to scripts/output/ (or shown interactively with --show):

  1. usl_curves.png          — USL throughput X(N) for AI agents and human teams
  2. beta_eff_vs_cg.png      — β_eff = β₀×(1−CG) coupling; how CG shifts reconciliation cost
  3. n_max_vs_cg.png         — N_max ceiling as a function of CG_mean
  4. cjt_quality.png         — CJT ensemble quality p_ensemble(N) at varying error correlation ρ
  5. n_eff_vs_rho.png        — EigenCalibration: N_eff participation ratio vs scalar ρ
  6. talagrand_histograms.png — Rank histogram shapes (flat=calibrated, U=over-confident, Λ=under-confident)

Usage:
    python scripts/simulate.py              # saves PNGs to scripts/output/
    python scripts/simulate.py --show       # opens interactive matplotlib window

Requires: numpy, matplotlib  (pre-installed in devcontainer)
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# ---------------------------------------------------------------------------
# Calibration constants (must match docs/architecture/math.md §3 table)
# ---------------------------------------------------------------------------
TIERS = {
    "AI agents":    {"alpha": 0.15, "beta0": 0.039, "cg": 0.40},
    "Human teams":  {"alpha": 0.10, "beta0": 0.0225, "cg": 0.60},
}


# ---------------------------------------------------------------------------
# Core formulas
# ---------------------------------------------------------------------------

def beta_eff(beta0: float, cg: float) -> float:
    """β_eff = β₀ × (1 − CG_mean)  — higher CG → cheaper reconciliation."""
    return max(beta0 * (1.0 - cg), 1e-9)


def usl(n: int, alpha: float, beta: float) -> float:
    """Universal Scalability Law: X(N) = N / (1 + α(N−1) + β·N(N−1))."""
    return n / (1.0 + alpha * (n - 1) + beta * n * (n - 1))


def n_max(alpha: float, beta: float) -> float:
    """Throughput-maximising agent count: √((1−α)/β)."""
    return math.sqrt((1.0 - alpha) / beta)


def cjt_ensemble(p: float, n: int, rho: float) -> float:
    """
    CJT ensemble accuracy with pairwise error correlation ρ.
    p_ensemble = p + (1−p) × [1 − (1−p)^((n-1)/2)] × correction
    Approximation from Boland (1989): majority vote with correlation.
    """
    if n == 1:
        return p
    # Effective independent votes shrink with correlation
    n_eff_votes = 1.0 + (n - 1) * (1.0 - rho)
    # Majority probability via normal approximation
    mean = n * p
    var = n * p * (1.0 - p) + n * (n - 1) * rho * p * (1.0 - p)
    std = math.sqrt(max(var, 1e-12))
    threshold = n / 2.0
    # Use error function for CDF of normal
    z = (mean - threshold) / std
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def participation_ratio(rho: float, m: int) -> float:
    """
    N_eff = 1/∑λᵢ² where λᵢ are eigenvalues of a uniform-correlation matrix.
    For m×m matrix with off-diagonal ρ: eigenvalues are (1+(m-1)ρ) once and (1-ρ) m-1 times.
    """
    lam1 = 1.0 + (m - 1) * rho
    lam2 = 1.0 - rho
    total = lam1**2 + (m - 1) * lam2**2
    trace_sq = lam1 + (m - 1) * lam2  # = m (trace of identity-scaled)
    return (trace_sq**2) / (m * total)


# ---------------------------------------------------------------------------
# Plot 1: USL throughput curves
# ---------------------------------------------------------------------------

def plot_usl_curves(show: bool) -> None:
    ns = np.arange(1, 20)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("USL Throughput Curves  X(N) = N / (1 + α(N−1) + β_eff·N(N−1))")

    for ax, (label, tier) in zip(axes, TIERS.items()):
        alpha = tier["alpha"]
        b0 = tier["beta0"]
        cg = tier["cg"]
        be = beta_eff(b0, cg)
        nm = n_max(alpha, be)

        xs = [usl(n, alpha, be) for n in ns]
        ax.plot(ns, xs, "b-o", markersize=4, label=f"β_eff={be:.4f}")
        ax.axvline(nm, color="r", linestyle="--", label=f"N_max={nm:.1f}")
        ax.set_title(label)
        ax.set_xlabel("N (agents)")
        ax.set_ylabel("Relative throughput X(N)")
        ax.legend(fontsize=9)
        ax.set_xlim(1, 19)
        ax.set_ylim(0, None)
        ax.text(
            0.98, 0.05,
            f"α={alpha}, β₀={b0}, CG={cg}",
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="gray",
        )

    plt.tight_layout()
    _save_or_show("usl_curves.png", show)


# ---------------------------------------------------------------------------
# Plot 2: β_eff vs CG coupling
# ---------------------------------------------------------------------------

def plot_beta_eff_vs_cg(show: bool) -> None:
    cg_range = np.linspace(0.0, 1.0, 200)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, tier in TIERS.items():
        b0 = tier["beta0"]
        be_vals = [beta_eff(b0, cg) for cg in cg_range]
        ax.plot(cg_range, be_vals, label=f"{label} (β₀={b0})")
        cg_op = tier["cg"]
        be_op = beta_eff(b0, cg_op)
        ax.scatter([cg_op], [be_op], zorder=5, s=60)
        ax.annotate(
            f"  operating point\n  CG={cg_op}, β_eff={be_op:.4f}",
            xy=(cg_op, be_op), fontsize=8,
        )

    ax.set_xlabel("CG_mean  (embedding agreement rate)")
    ax.set_ylabel("β_eff  (pairwise reconciliation cost)")
    ax.set_title("β_eff = β₀ × (1 − CG_mean)\nHigher CG → cheaper reconciliation → higher N_max")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    plt.tight_layout()
    _save_or_show("beta_eff_vs_cg.png", show)


# ---------------------------------------------------------------------------
# Plot 3: N_max vs CG
# ---------------------------------------------------------------------------

def plot_n_max_vs_cg(show: bool) -> None:
    cg_range = np.linspace(0.01, 0.99, 300)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, tier in TIERS.items():
        alpha = tier["alpha"]
        b0 = tier["beta0"]
        nm_vals = [n_max(alpha, beta_eff(b0, cg)) for cg in cg_range]
        ax.plot(cg_range, nm_vals, label=label)
        cg_op = tier["cg"]
        nm_op = n_max(alpha, beta_eff(b0, cg_op))
        ax.scatter([cg_op], [nm_op], zorder=5, s=60)
        ax.annotate(
            f"  N_max={nm_op:.1f}",
            xy=(cg_op, nm_op), fontsize=8,
        )

    ax.set_xlabel("CG_mean  (embedding agreement rate)")
    ax.set_ylabel("N_max  (throughput-maximising agent count)")
    ax.set_title("N_max = √((1−α) / β_eff)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    plt.tight_layout()
    _save_or_show("n_max_vs_cg.png", show)


# ---------------------------------------------------------------------------
# Plot 4: CJT quality at varying ρ
# ---------------------------------------------------------------------------

def plot_cjt_quality(show: bool) -> None:
    ns = list(range(1, 13))
    p_agent = 0.70  # individual agent accuracy
    rho_values = [0.0, 0.2, 0.4, 0.6, 0.8]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(rho_values)))

    for rho, color in zip(rho_values, colors):
        qs = [cjt_ensemble(p_agent, n, rho) for n in ns]
        ax.plot(ns, qs, "o-", color=color, label=f"ρ={rho:.1f}", markersize=4)

    ax.axhline(p_agent, color="gray", linestyle=":", label=f"Single agent p={p_agent}")
    ax.set_xlabel("N (agents)")
    ax.set_ylabel("p_ensemble  (majority-correct probability)")
    ax.set_title(
        f"CJT Ensemble Quality  (p={p_agent}, varying error correlation ρ)\n"
        "High ρ kills the Condorcet gain — diversity matters"
    )
    ax.legend(fontsize=9)
    ax.set_xlim(1, 12)
    ax.set_ylim(p_agent - 0.05, 1.0)
    plt.tight_layout()
    _save_or_show("cjt_quality.png", show)


# ---------------------------------------------------------------------------
# Plot 5: N_eff eigenvalue vs scalar ρ
# ---------------------------------------------------------------------------

def plot_n_eff_vs_rho(show: bool) -> None:
    rho_range = np.linspace(0.0, 0.99, 300)
    m_values = [3, 6, 9, 12]

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in m_values:
        n_eff_vals = [participation_ratio(rho, m) for rho in rho_range]
        ax.plot(rho_range, n_eff_vals, label=f"m={m}")

    ax.set_xlabel("ρ  (uniform pairwise correlation)")
    ax.set_ylabel("N_eff  (participation ratio)")
    ax.set_title(
        "EigenCalibration: N_eff vs scalar ρ\n"
        "N_eff → 1 as ρ → 1 (all adapters redundant); N_eff → m as ρ → 0 (fully diverse)"
    )
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    plt.tight_layout()
    _save_or_show("n_eff_vs_rho.png", show)


# ---------------------------------------------------------------------------
# Plot 6: Talagrand rank histogram shapes
# ---------------------------------------------------------------------------

def plot_talagrand_histograms(show: bool) -> None:
    bins = np.arange(0, 11)
    n_bins = 10

    # Flat (calibrated): uniform distribution
    flat = np.ones(n_bins) / n_bins

    # U-shape (over-confident): mass at tails
    u_raw = np.array([3, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 3], dtype=float)
    u_shape = u_raw / u_raw.sum()

    # Λ-shape (under-confident): mass at centre
    lam_raw = np.array([0.5, 1, 2, 3, 4, 4, 3, 2, 1, 0.5], dtype=float)
    lam_shape = lam_raw / lam_raw.sum()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Talagrand Rank Histogram Shapes")

    shapes = [
        ("Flat (calibrated)\n→ no τ adjustment", flat, "steelblue"),
        ("U-shape (over-confident)\n→ increase τ_spread +20%", u_shape, "tomato"),
        ("Λ-shape (under-confident)\n→ DiversityWarning emitted", lam_shape, "goldenrod"),
    ]
    for ax, (title, vals, color) in zip(axes, shapes):
        ax.bar(np.arange(1, n_bins + 1), vals, color=color, edgecolor="white", width=0.85)
        ax.axhline(1.0 / n_bins, color="gray", linestyle="--", linewidth=1, label="uniform")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Rank bin")
        ax.set_ylabel("Frequency")
        ax.set_xlim(0.5, n_bins + 0.5)
        ax.set_ylim(0, max(vals) * 1.2)

    plt.tight_layout()
    _save_or_show("talagrand_histograms.png", show)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_or_show(filename: str, show: bool) -> None:
    if show:
        plt.show()
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved: {path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="H2AI visualization suite")
    parser.add_argument("--show", action="store_true", help="Open interactive window instead of saving")
    args = parser.parse_args()

    print("H2AI simulate.py — generating plots...")
    plot_usl_curves(args.show)
    plot_beta_eff_vs_cg(args.show)
    plot_n_max_vs_cg(args.show)
    plot_cjt_quality(args.show)
    plot_n_eff_vs_rho(args.show)
    plot_talagrand_histograms(args.show)
    if not args.show:
        print(f"\nDone. PNGs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
