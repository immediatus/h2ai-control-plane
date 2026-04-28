"""
USL simulation and visualisation — docs/architecture/math-apparatus.md

Produces seven plots saved to scripts/output/ (or shown interactively with --show):
  1. 01_usl_three_layers.png      — USL throughput curves for all three calibrated layers
  2. 02_cg_mean_effect.png        — Effect of CG_mean on N_max; how constraint corpus quality shifts the ceiling
  3. 03_pareto_matrix.png         — Topology Pareto matrix heatmap (7 topologies × 3 axes)
  4. 04_dark_knowledge_gap.png    — J_eff distribution and task acceptance rate vs constraint corpus size
  5. 05_tao_error_reduction.png   — TAO loop c_i decay curves; BFT threshold crossover turns (Def 11)
  6. 06_harness_attribution.png   — Stacked attribution decomposition: baseline + topology + verify + TAO (Def 13)
  7. 07_attribution_sensitivity.png — Q_total monotonicity in N, TAO turns, filter_ratio (Prop 8)

WHEN TO RUN
-----------
Run when exploring theory or verifying a parameter change visually. The validator
(validate_math.py) gives a binary pass/fail; this script shows the shape of the
equations — useful when changing calibration constants or topology scores.

    python scripts/simulate_usl.py              # saves PNGs to scripts/output/
    python scripts/simulate_usl.py --show       # opens interactive matplotlib window

Requires: numpy, matplotlib  (pre-installed in devcontainer via pip)

CROSS-REFERENCE MAP  (doc section → constants/functions/plots in this file)
----------------------------------------------------------------------------
§ Definition 2  (USL formula)   — usl()                          → Plots 1, 2
§ Definition 3  (CG)            — beta_eff() uses CG_mean        → Plot 2
§ Definition 4  (β_eff)         — beta_eff(beta_base, cg)        → Plots 1, 2
§ Definition 5  (Extended USL)  — usl() + beta_eff() compose     → Plot 2 right panel
§ Definition 9  (Pareto axes)   — TOPOLOGIES (T, E, D tuples)    → Plot 3
§ Definition 10 (J_eff gate)    — J_EFF_GATE = 0.4              → Plot 4
§ Definition 11 (TAO reduction) — tao_c_i_effective()            → Plot 5
§ Definition 12 (verify gain)   — verification_gain()            → Plots 6, 7
§ Definition 13 (attribution)   — harness_attribution()          → Plots 6, 7
§ Proposition 1 (N_max)         — n_max()                        → Plots 1, 2, 6, 7
§ Proposition 5 (frontier)      — TOPOLOGIES frontier flags      → Plot 3
§ Proposition 8 (monotonicity)  — harness_attribution()          → Plot 7
§ §3 Calibration table          — LAYERS (must match doc table)  → Plots 1, 2

CONSTANTS THAT MUST STAY IN SYNC WITH THE DOC
----------------------------------------------
    J_EFF_GATE    = 0.4   → docs/architecture/math-apparatus.md §4 (J_eff gate row)
    BFT_THRESHOLD = 0.85  → docs/architecture/math-apparatus.md §Proposition 5 + Def 11 crossover
    TAO_DECAY     = 0.60  → docs/architecture/math-apparatus.md Definition 11 (r_tao = 0.40)
    LAYERS              → §3 Calibration and §Proposition 1 calibrated ceilings table
                           (must also match CALIBRATION_TABLE in validate_math.py)
    TOPOLOGIES          → docs/guides/theory-to-implementation.md Pareto Summary table
"""

import math
import sys
import os

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except ImportError:
    print("ERROR: numpy and matplotlib are required.")
    print("       pip install numpy matplotlib")
    sys.exit(1)

SHOW = "--show" in sys.argv
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Core functions (mirror validate_math.py — intentionally self-contained)
# ---------------------------------------------------------------------------

def usl(N: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return N / (1 + alpha * (N - 1) + beta * N * (N - 1))


def n_max(alpha: float, beta: float) -> float:
    return math.sqrt((1 - alpha) / beta)


def beta_eff(beta_base: float, cg_mean: float) -> float:
    return beta_base / cg_mean


def save_or_show(fig: plt.Figure, filename: str) -> None:
    if SHOW:
        plt.show()
    else:
        path = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1 — USL throughput curves for three calibrated layers
# ---------------------------------------------------------------------------

LAYERS = [
    # (label, alpha, beta_base, cg_mean, color)
    ("CPU cores  (α=0.02, β_eff=0.0003, N_max≈57)",  0.02, 0.0003, 1.0, "#2563eb"),
    ("Human teams (α=0.10, β_eff=0.0083, N_max≈10)", 0.10, 0.005,  0.6, "#16a34a"),
    ("AI agents  (α=0.15, β_eff=0.025,  N_max≈6)",   0.15, 0.01,   0.4, "#dc2626"),
]

# § Definition 2, Definition 5, Proposition 1, §3 Calibration
# Visually confirms: X(1)=1, peaks at N_max, retrograde past peak, three-layer gap
print("\nPlot 1 — USL throughput curves")
fig, ax = plt.subplots(figsize=(10, 6))

N = np.linspace(1, 70, 500)

for label, alpha, bb, cg, color in LAYERS:
    be = beta_eff(bb, cg)
    X = usl(N, alpha, be)
    nm = n_max(alpha, be)
    X_peak = usl(np.array([nm]), alpha, be)[0]

    ax.plot(N, X, color=color, linewidth=2, label=label)
    ax.axvline(nm, color=color, linewidth=1, linestyle="--", alpha=0.5)
    ax.annotate(
        f"N_max={nm:.0f}",
        xy=(nm, X_peak),
        xytext=(nm + 1.5, X_peak - 0.3),
        fontsize=8,
        color=color,
        arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
    )

ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
ax.fill_between(N, 0, 0.05, alpha=0.05, color="red", label="Retrograde region")
ax.set_xlabel("Number of agents / cores / team members (N)", fontsize=11)
ax.set_ylabel("Normalised throughput X(N)", fontsize=11)
ax.set_title("Universal Scalability Law — Three Calibrated Layers\n"
             "Dashed verticals mark N_max; throughput falls into retrograde past each peak",
             fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(1, 70)
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_or_show(fig, "01_usl_three_layers.png")


# ---------------------------------------------------------------------------
# Plot 2 — Effect of CG_mean on N_max (AI-agent layer)
# ---------------------------------------------------------------------------

# § Definition 3, Definition 4, Proposition 1
# Visually confirms: higher CG_mean → lower β_eff → higher N_max (better constraint corpus = more agents)
print("Plot 2 — CG_mean vs N_max for AI-agent layer")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cg_values = np.linspace(0.2, 1.0, 200)
alpha_ai = 0.15
bb_ai = 0.01

n_max_values = [n_max(alpha_ai, beta_eff(bb_ai, cg)) for cg in cg_values]

ax = axes[0]
ax.plot(cg_values, n_max_values, color="#7c3aed", linewidth=2)
ax.axhline(6,  color="#dc2626", linestyle="--", linewidth=1, label="CG_mean=0.4 (typical AI, N_max≈6)")
ax.axhline(10, color="#16a34a", linestyle="--", linewidth=1, label="CG_mean=0.6 (human team, N_max≈10)")
ax.axvline(0.4, color="#dc2626", linestyle=":", linewidth=0.8, alpha=0.6)
ax.axvline(0.6, color="#16a34a", linestyle=":", linewidth=0.8, alpha=0.6)
ax.set_xlabel("CG_mean (mean common ground across agent pairs)", fontsize=10)
ax.set_ylabel("N_max (scalability ceiling)", fontsize=10)
ax.set_title("Higher common ground → higher N_max\n"
             "Better constraint corpus and τ alignment raises the ceiling", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.2, 1.0)

# Right panel: USL curves at three CG_mean values
ax2 = axes[1]
N_short = np.linspace(1, 20, 200)
for cg, color, label in [
    (0.4, "#dc2626", "CG_mean=0.40 (typical AI)"),
    (0.6, "#f97316", "CG_mean=0.60 (with constraint corpus)"),
    (0.9, "#16a34a", "CG_mean=0.90 (high alignment)"),
]:
    be = beta_eff(bb_ai, cg)
    X = usl(N_short, alpha_ai, be)
    nm = n_max(alpha_ai, be)
    ax2.plot(N_short, X, color=color, linewidth=2, label=f"{label}  N_max≈{nm:.0f}")
    ax2.axvline(nm, color=color, linewidth=1, linestyle="--", alpha=0.4)

ax2.set_xlabel("Number of AI agents (N)", fontsize=10)
ax2.set_ylabel("Normalised throughput X(N)", fontsize=10)
ax2.set_title("USL curves at different CG_mean values\n"
              "Constraint corpus quality directly shifts the scalability ceiling", fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, 20)

fig.suptitle("Effect of Common Ground on Scalability  (AI-agent layer, α=0.15, β_base=0.01)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save_or_show(fig, "02_cg_mean_effect.png")


# ---------------------------------------------------------------------------
# Plot 3 — Topology Pareto matrix heatmap
# ---------------------------------------------------------------------------

# § Definition 9 (Pareto axes), Proposition 5 (frontier claim)
# TOPOLOGIES scores must match docs/guides/theory-to-implementation.md Pareto Summary table
print("Plot 3 — Topology Pareto matrix heatmap")

TOPOLOGIES = [
    # (name, T, E, D, frontier)
    ("Hierarchical Tree", 0.96, 0.96, 0.60, True),
    ("Team-Swarm Hybrid", 0.84, 0.91, 0.95, True),
    ("Ensemble + CRDT",   0.84, 0.84, 0.90, True),
    ("Star",              0.52, 0.78, 0.75, False),
    ("Oracle",            0.50, 0.88, 0.20, False),
    ("Pipeline",          0.48, 0.18, 0.20, False),
    ("Flat Panel",        0.18, 0.18, 0.90, False),
]

labels = [t[0] for t in TOPOLOGIES]
scores = np.array([[t[1], t[2], t[3]] for t in TOPOLOGIES])
frontier = [t[4] for t in TOPOLOGIES]
axes_labels = ["T  Throughput", "E  Containment", "D  Diversity"]

fig, ax = plt.subplots(figsize=(9, 5))

cmap = mcolors.LinearSegmentedColormap.from_list(
    "rd_gn", ["#dc2626", "#fbbf24", "#16a34a"]
)

im = ax.imshow(scores, cmap=cmap, vmin=0, vmax=1, aspect="auto")

# Cell text
for i in range(len(TOPOLOGIES)):
    for j in range(3):
        val = scores[i, j]
        text_color = "white" if val < 0.35 or val > 0.80 else "black"
        ax.text(j, i, f"{val*100:.0f}%", ha="center", va="center",
                fontsize=11, fontweight="bold", color=text_color)

# Frontier indicator
for i, is_frontier in enumerate(frontier):
    if is_frontier:
        ax.add_patch(plt.Rectangle((-0.5, i - 0.5), 3, 1,
                                   fill=False, edgecolor="#16a34a",
                                   linewidth=2.5, zorder=3))

# Dashed separator after frontier topologies
frontier_count = sum(frontier)
ax.axhline(frontier_count - 0.5, color="gray", linewidth=1.5,
           linestyle="--", alpha=0.7)

ax.set_xticks(range(3))
ax.set_xticklabels(axes_labels, fontsize=10, fontweight="bold")
ax.set_yticks(range(len(TOPOLOGIES)))
ax.set_yticklabels([
    f"★ {l}" if f else f"  {l}"
    for l, f in zip(labels, frontier)
], fontsize=10)
ax.set_title("H2AI Topology Pareto Matrix\n"
             "★ = Pareto frontier  |  green border = non-dominated  |  dashed line = frontier boundary",
             fontsize=10)
plt.colorbar(im, ax=ax, label="Score (0 = worst, 1 = best)", shrink=0.8)
fig.tight_layout()
save_or_show(fig, "03_pareto_matrix.png")


# ---------------------------------------------------------------------------
# Plot 4 — Dark Knowledge Gap: J_eff vs acceptance
# ---------------------------------------------------------------------------

# § Definition 10 (J_eff gate = 0.4)
# J_EFF_GATE must match docs/architecture/math-apparatus.md §4 and validate_math.py
print("Plot 4 — Dark Knowledge Gap and J_eff gate")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: J_eff distribution across a corpus with increasing ADR count
np.random.seed(42)
J_EFF_GATE = 0.4
adr_counts = [0, 1, 2, 3, 5, 7]
acceptance_rates = []
medians = []

ax = axes[0]
positions = []
data_for_box = []

for i, n_adrs in enumerate(adr_counts):
    # Simulate J_eff values: more ADRs → higher mean and tighter distribution
    mean_j = min(0.15 + n_adrs * 0.09, 0.85)
    std_j  = max(0.18 - n_adrs * 0.02, 0.05)
    samples = np.clip(np.random.normal(mean_j, std_j, 500), 0, 1)
    data_for_box.append(samples)
    positions.append(i)
    acceptance_rates.append((samples >= J_EFF_GATE).mean())
    medians.append(np.median(samples))

bp = ax.boxplot(data_for_box, positions=positions, widths=0.6,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=2))

colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(adr_counts)))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.axhline(J_EFF_GATE, color="#dc2626", linewidth=1.5, linestyle="--",
           label=f"J_eff gate = {J_EFF_GATE}")
ax.set_xticks(positions)
ax.set_xticklabels([str(n) for n in adr_counts])
ax.set_xlabel("Number of ADRs in corpus", fontsize=10)
ax.set_ylabel("J_eff (Dark Knowledge Gap)", fontsize=10)
ax.set_title("J_eff distribution vs constraint corpus size\n"
             "Tasks below the gate return ContextUnderflowError", fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
ax.set_ylim(0, 1)

# Right: acceptance rate vs ADR count
ax2 = axes[1]
ax2.bar(adr_counts, [r * 100 for r in acceptance_rates],
        color=plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(adr_counts))),
        edgecolor="gray", linewidth=0.5)
ax2.axhline(100, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
for i, (n, rate) in enumerate(zip(adr_counts, acceptance_rates)):
    ax2.text(n, rate * 100 + 1.5, f"{rate*100:.0f}%", ha="center",
             fontsize=9, fontweight="bold")
ax2.set_xlabel("Number of ADRs in corpus", fontsize=10)
ax2.set_ylabel("Task acceptance rate (%)", fontsize=10)
ax2.set_title("Task acceptance rate vs constraint corpus size\n"
              "Empty corpus → most tasks rejected; 5+ docs → near-full acceptance", fontsize=10)
ax2.set_ylim(0, 110)
ax2.set_xticks(adr_counts)
ax2.grid(True, axis="y", alpha=0.3)

fig.suptitle("Dark Knowledge Gap (J_eff) — Effect of Constraint Corpus on Task Acceptance",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save_or_show(fig, "04_dark_knowledge_gap.png")

# ---------------------------------------------------------------------------
# Harness extension formulas (Definitions 11–13, Proposition 8)
# ---------------------------------------------------------------------------

BFT_THRESHOLD = 0.85   # must match validate_math.py and math-apparatus.md §Prop 5
TAO_DECAY     = 0.60   # (1 - r_tao) = 0.60; r_tao = 0.40 per Definition 11


def tao_c_i_effective(c_i: float, turns: int) -> float:
    """Definition 11: c_i_effective(t) = c_i × 0.60^(t-1)"""
    return c_i * (TAO_DECAY ** (turns - 1))


def topology_gain(c_i: float, n: int, alpha: float, beta_e: float) -> float:
    """Definition 13 G_topology component.

    Uses actual USL throughput X(N) = N / (1 + α(N-1) + βN(N-1)).
    At N=1: X=1 → gain=0 (by definition, single agent = no topology benefit).
    """
    n = max(n, 1)
    usl_n = n / (1.0 + alpha * (n - 1) + beta_e * n * (n - 1))
    usl_n = max(usl_n, 1.0)  # N=1 gives exactly 1.0
    return max(c_i * (1.0 - 1.0 / usl_n), 0.0)


def tao_gain(c_i: float, turns: float) -> float:
    """Definition 13 G_tao component."""
    baseline = 1.0 - c_i
    tao_c = c_i * (TAO_DECAY ** (turns - 1))
    return max((1.0 - tao_c) - baseline, 0.0)


def verification_gain(c_i: float, filter_ratio: float) -> float:
    """Definition 12 / 13 G_verification component."""
    return c_i * (1.0 - filter_ratio)


def harness_attribution(c_i: float, n: int, alpha: float, beta_e: float,
                         filter_ratio: float, tao_turns: float) -> dict:
    """Definition 13: full Q_total decomposition."""
    baseline = 1.0 - c_i
    g_topo   = topology_gain(c_i, n, alpha, beta_e)
    g_tao    = tao_gain(c_i, tao_turns)
    g_verify = verification_gain(c_i, filter_ratio)
    total    = min(baseline + g_topo + g_tao + g_verify, 1.0)
    return dict(baseline=baseline, topology=g_topo, tao=g_tao,
                verification=g_verify, total=total)


# ---------------------------------------------------------------------------
# Plot 5 — TAO Error Reduction (Definition 11)
# ---------------------------------------------------------------------------

# § Definition 11 — c_i_effective(t) = c_i × 0.60^(t-1)
# Shows: at which turn does each agent class drop below BFT threshold?
# Operational use: operator sets max_turns to ensure Shell agents avoid BFT path.
print("Plot 5 — TAO error reduction and BFT threshold crossover")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

turns = np.arange(1, 5)
CI_VALUES = [
    (0.30, "#16a34a",  "Evaluator / Synthesizer  (c_i = 0.30)"),
    (0.50, "#f97316",  "Executor                 (c_i = 0.50)"),
    (0.70, "#dc8c00",  "Swarm Coordinator        (c_i = 0.70)"),
    (0.90, "#dc2626",  "Shell / Auditor agent    (c_i = 0.90)"),
]

ax = axes[0]
for c_i, color, label in CI_VALUES:
    eff = [tao_c_i_effective(c_i, int(t)) for t in turns]
    ax.plot(turns, eff, color=color, linewidth=2.5, marker="o", markersize=7, label=label)

    # Annotate BFT crossover point if it exists within t=1..4
    for t in range(1, 5):
        if tao_c_i_effective(c_i, t) <= BFT_THRESHOLD:
            ax.annotate(
                f"t={t}: {tao_c_i_effective(c_i, t):.2f}",
                xy=(t, tao_c_i_effective(c_i, t)),
                xytext=(t + 0.15, tao_c_i_effective(c_i, t) + 0.03),
                fontsize=7.5, color=color,
            )
            break

ax.axhline(BFT_THRESHOLD, color="#7c3aed", linewidth=1.5, linestyle="--",
           label=f"BFT threshold = {BFT_THRESHOLD}")
ax.fill_between([0.8, 4.2], BFT_THRESHOLD, 1.0, alpha=0.07, color="#dc2626",
                label="BFT-required zone")
ax.set_xlabel("TAO turns (t)", fontsize=10)
ax.set_ylabel("c_i_effective = c_i × 0.60^(t−1)", fontsize=10)
ax.set_title("TAO Error Reduction per Agent Class\n"
             "Purple dashed = BFT threshold — drops below → CRDT merge unlocked",
             fontsize=10)
ax.set_xticks(turns)
ax.set_xlim(0.8, 4.2)
ax.set_ylim(0.0, 1.02)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

# Right panel: reduction ratio vs c_i baseline for each turn count
ax2 = axes[1]
ci_range = np.linspace(0.0, 1.0, 200)
for t, color, lstyle in [(1, "#94a3b8", "-"), (2, "#f97316", "-"),
                          (3, "#2563eb", "-"), (4, "#16a34a", "-")]:
    eff = [tao_c_i_effective(c, t) for c in ci_range]
    ax2.plot(ci_range, eff, color=color, linewidth=2, linestyle=lstyle,
             label=f"t = {t}  (factor {TAO_DECAY**(t-1):.3f}×)")

ax2.axhline(BFT_THRESHOLD, color="#7c3aed", linewidth=1.5, linestyle="--",
            label=f"BFT threshold = {BFT_THRESHOLD}")
ax2.fill_between(ci_range, BFT_THRESHOLD, 1.0, alpha=0.07, color="#dc2626")
ax2.plot([0, 1], [0, 1], color="gray", linewidth=1, linestyle=":", alpha=0.5,
         label="no reduction (t=1 baseline)")
ax2.set_xlabel("Baseline c_i (role error cost before TAO)", fontsize=10)
ax2.set_ylabel("c_i_effective after t turns", fontsize=10)
ax2.set_title("Effective c_i vs Baseline — All TAO Turn Counts\n"
              "Agents above purple line force BFT; below → CRDT merge",
              fontsize=10)
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1.02)

fig.suptitle("Definition 11 — TAO Error Reduction: c_i_effective(t) = c_i × 0.60^(t−1)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save_or_show(fig, "05_tao_error_reduction.png")


# ---------------------------------------------------------------------------
# Plot 6 — Harness Attribution Decomposition (Definition 13)
# ---------------------------------------------------------------------------

# § Definition 13 — Q_total = Q_baseline + G_topology + G_verification + G_tao
# Shows: each harness component's contribution across four representative configs.
# Operational use: the enterprise pitch artifact — "the harness added X% quality".
print("Plot 6 — Harness attribution decomposition (stacked bar)")

# AI-agent calibration defaults
ALPHA_AI   = 0.15
BETA_AI    = 0.025   # beta_eff for AI agents (β_base/CG_mean = 0.01/0.4)

CONFIGS = [
    # (label, n_agents, tao_turns, filter_ratio)
    ("Single agent\n(no harness)",   1, 1, 1.0),
    ("4-agent ensemble\n(topology)", 4, 1, 1.0),
    ("Ensemble +\nTAO (t=3)",        4, 3, 1.0),
    ("Full harness\n(all components)", 4, 3, 0.5),
]

C_I_DEFAULT = 0.50   # Executor baseline

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

COMPONENT_COLORS = {
    "baseline":     "#64748b",
    "topology":     "#2563eb",
    "tao":          "#16a34a",
    "verification": "#f97316",
}
COMPONENT_LABELS = {
    "baseline":     "Q_baseline  (model, no harness)",
    "topology":     "G_topology  (N-agent USL ensemble)",
    "tao":          "G_tao       (TAO iterative refinement)",
    "verification": "G_verify    (Verification Phase filter)",
}

ax = axes[0]
config_labels = [c[0] for c in CONFIGS]
x = np.arange(len(CONFIGS))
width = 0.55

bottoms = np.zeros(len(CONFIGS))
for component in ["baseline", "topology", "tao", "verification"]:
    values = []
    for _, n, turns, fr in CONFIGS:
        attr = harness_attribution(C_I_DEFAULT, n, ALPHA_AI, BETA_AI, fr, turns)
        values.append(attr[component])
    bars = ax.bar(x, values, width, bottom=bottoms,
                  color=COMPONENT_COLORS[component],
                  label=COMPONENT_LABELS[component])
    bottoms += np.array(values)

# Annotate total Q_total on top of each bar
for i, (_, n, turns, fr) in enumerate(CONFIGS):
    attr = harness_attribution(C_I_DEFAULT, n, ALPHA_AI, BETA_AI, fr, turns)
    ax.text(i, attr["total"] + 0.01, f"{attr['total']*100:.0f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(config_labels, fontsize=9)
ax.set_ylabel("Quality score Q_total  (0 = worst, 1 = best)", fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_title(f"Harness Quality Attribution — Executor agent (c_i = {C_I_DEFAULT})\n"
             "AI layer: α=0.15, β_eff=0.025",
             fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, axis="y", alpha=0.3)

# Right panel: same breakdown for Shell agent (c_i=0.9) to show BFT interaction
C_I_SHELL = 0.90
ax2 = axes[1]
bottoms = np.zeros(len(CONFIGS))
for component in ["baseline", "topology", "tao", "verification"]:
    values = []
    for _, n, turns, fr in CONFIGS:
        attr = harness_attribution(C_I_SHELL, n, ALPHA_AI, BETA_AI, fr, turns)
        values.append(attr[component])
    ax2.bar(x, values, width, bottom=bottoms,
            color=COMPONENT_COLORS[component],
            label=COMPONENT_LABELS[component])
    bottoms += np.array(values)

for i, (_, n, turns, fr) in enumerate(CONFIGS):
    attr = harness_attribution(C_I_SHELL, n, ALPHA_AI, BETA_AI, fr, turns)
    ax2.text(i, attr["total"] + 0.01, f"{attr['total']*100:.0f}%",
             ha="center", va="bottom", fontsize=11, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(config_labels, fontsize=9)
ax2.set_ylabel("Quality score Q_total", fontsize=10)
ax2.set_ylim(0, 1.12)
ax2.set_title(f"Harness Quality Attribution — Shell agent (c_i = {C_I_SHELL})\n"
              "Higher baseline error → harness gain is larger in absolute terms",
              fontsize=10)
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(True, axis="y", alpha=0.3)

fig.suptitle("Definition 13 — Harness Attribution Decomposition: "
             "Q_total = Q_baseline + G_topology + G_verify + G_tao",
             fontsize=11, fontweight="bold")
fig.tight_layout()
save_or_show(fig, "06_harness_attribution.png")


# ---------------------------------------------------------------------------
# Plot 7 — Attribution Sensitivity / Proposition 8 Monotonicity
# ---------------------------------------------------------------------------

# § Proposition 8 — Q_total is monotone in N (≤ N_max), TAO turns, and (1−filter_ratio).
# Three sensitivity curves sweeping one parameter while holding the others constant.
# All curves must be strictly non-decreasing — any dip would falsify Proposition 8.
print("Plot 7 — Attribution sensitivity / Proposition 8 monotonicity")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

C_I_BASE   = 0.50
N_MAX_AI   = int(n_max(ALPHA_AI, BETA_AI))   # ≈ 6

# Panel A: Q_total vs N (1..N_max) — topology gain
ax = axes[0]
ns = np.arange(1, N_MAX_AI + 1)
q_vs_n = [harness_attribution(C_I_BASE, int(n_v), ALPHA_AI, BETA_AI,
                               filter_ratio=1.0, tao_turns=1)["total"]
          for n_v in ns]
ax.plot(ns, q_vs_n, color="#2563eb", linewidth=2.5, marker="o", markersize=8)
ax.fill_between(ns, harness_attribution(C_I_BASE, 1, ALPHA_AI, BETA_AI, 1.0, 1)["total"],
                q_vs_n, alpha=0.12, color="#2563eb", label="topology gain")
ax.set_xlabel("Number of agents N  (≤ N_max)", fontsize=10)
ax.set_ylabel("Q_total", fontsize=10)
ax.set_title(f"Monotone in N  (N_max = {N_MAX_AI}, c_i={C_I_BASE})\n"
             "Each additional agent within N_max raises Q_total",
             fontsize=10)
ax.set_xticks(ns)
ax.set_xlim(0.5, N_MAX_AI + 0.5)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# Panel B: Q_total vs TAO turns (1..4) — tao gain
ax2 = axes[1]
tao_range = np.arange(1, 5)
q_vs_tao = [harness_attribution(C_I_BASE, 4, ALPHA_AI, BETA_AI,
                                 filter_ratio=1.0, tao_turns=float(t))["total"]
            for t in tao_range]
ax2.plot(tao_range, q_vs_tao, color="#16a34a", linewidth=2.5, marker="s", markersize=8)
ax2.fill_between(tao_range, q_vs_tao[0], q_vs_tao, alpha=0.12, color="#16a34a",
                 label="TAO gain")
ax2.set_xlabel("TAO turns (t)", fontsize=10)
ax2.set_ylabel("Q_total", fontsize=10)
ax2.set_title(f"Monotone in TAO turns  (N=4, c_i={C_I_BASE})\n"
              "Geometric decay of c_i raises Q with diminishing returns",
              fontsize=10)
ax2.set_xticks(tao_range)
ax2.set_xlim(0.5, 4.5)
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3)

# Panel C: Q_total vs (1-filter_ratio) — verification strictness
ax3 = axes[2]
fr_range = np.linspace(0.0, 1.0, 50)   # filter_ratio: 1.0=no filter → 0.0=all filtered
q_vs_fr  = [harness_attribution(C_I_BASE, 4, ALPHA_AI, BETA_AI,
                                 filter_ratio=float(fr), tao_turns=1.0)["total"]
            for fr in fr_range]
strictness = 1.0 - fr_range   # x-axis: strictness = 1 - filter_ratio
ax3.plot(strictness, q_vs_fr, color="#f97316", linewidth=2.5)
ax3.fill_between(strictness, q_vs_fr[0], q_vs_fr, alpha=0.12, color="#f97316",
                 label="verification gain")
ax3.set_xlabel("Verification strictness  (1 − filter_ratio)", fontsize=10)
ax3.set_ylabel("Q_total", fontsize=10)
ax3.set_title(f"Monotone in verification strictness  (N=4, c_i={C_I_BASE})\n"
              "Stricter threshold removes more low-quality proposals",
              fontsize=10)
ax3.set_xlim(-0.02, 1.02)
ax3.set_ylim(0, 1.05)
ax3.grid(True, alpha=0.3)

# Print numeric findings to stdout for doc update
print("\n  ── Proposition 8 numeric findings ──")
print(f"  Q_total(N=1) = {q_vs_n[0]:.4f}   Q_total(N={N_MAX_AI}) = {q_vs_n[-1]:.4f}"
      f"   Δ(topology) = {q_vs_n[-1]-q_vs_n[0]:.4f}")
print(f"  Q_total(t=1) = {q_vs_tao[0]:.4f}   Q_total(t=4) = {q_vs_tao[-1]:.4f}"
      f"   Δ(TAO)      = {q_vs_tao[-1]-q_vs_tao[0]:.4f}")
q_fr_min = harness_attribution(C_I_BASE, 4, ALPHA_AI, BETA_AI, 1.0, 1.0)["total"]
q_fr_max = harness_attribution(C_I_BASE, 4, ALPHA_AI, BETA_AI, 0.0, 1.0)["total"]
print(f"  Q_total(fr=1.0) = {q_fr_min:.4f}   Q_total(fr=0.0) = {q_fr_max:.4f}"
      f"   Δ(verify)   = {q_fr_max-q_fr_min:.4f}")

# Verify monotonicity assertions (must all pass)
assert all(q_vs_n[i] <= q_vs_n[i+1] for i in range(len(q_vs_n)-1)), \
    "FAILED: Q_total not monotone in N"
assert all(q_vs_tao[i] <= q_vs_tao[i+1] for i in range(len(q_vs_tao)-1)), \
    "FAILED: Q_total not monotone in TAO turns"
assert all(q_vs_fr[i] >= q_vs_fr[i+1] for i in range(len(q_vs_fr)-1)), \
    "FAILED: Q_total not monotone in filter_ratio (should decrease as strictness increases)"
print("  ✓ Proposition 8 monotonicity verified for all three parameters")

fig.suptitle("Proposition 8 — Attribution Monotonicity: ∂Q/∂N ≥ 0,  ∂Q/∂t ≥ 0,  ∂Q/∂(1−f) ≥ 0",
             fontsize=11, fontweight="bold")
fig.tight_layout()
save_or_show(fig, "07_attribution_sensitivity.png")


# ---------------------------------------------------------------------------
# Print key numeric findings for doc cross-reference
# ---------------------------------------------------------------------------

print("\n  ── Key findings from simulation ──")

# TAO crossover turns by agent class
print("\n  TAO BFT crossover turns (c_i → drops below 0.85 at turn t):")
for c_i, _, label in CI_VALUES:
    for t in range(1, 6):
        if tao_c_i_effective(c_i, t) <= BFT_THRESHOLD:
            print(f"    {label.split('(')[0].strip():30s}  c_i={c_i}  → t={t}  "
                  f"(c_i_eff={tao_c_i_effective(c_i, t):.3f})")
            break
    else:
        print(f"    {label.split('(')[0].strip():30s}  c_i={c_i}  → never crosses BFT threshold")

# Attribution for each config (Executor)
print(f"\n  Harness attribution — Executor (c_i={C_I_DEFAULT}, AI layer α={ALPHA_AI}, β_eff={BETA_AI}):")
for label, n, turns, fr in CONFIGS:
    attr = harness_attribution(C_I_DEFAULT, n, ALPHA_AI, BETA_AI, fr, turns)
    print(f"    {label.replace(chr(10), ' '):35s}  Q_total={attr['total']*100:.0f}%"
          f"  (base={attr['baseline']*100:.0f}%"
          f"  topo=+{attr['topology']*100:.0f}%"
          f"  tao=+{attr['tao']*100:.0f}%"
          f"  verify=+{attr['verification']*100:.0f}%)")

# Attribution for Shell agent
print(f"\n  Harness attribution — Shell agent (c_i={C_I_SHELL}, AI layer):")
for label, n, turns, fr in CONFIGS:
    attr = harness_attribution(C_I_SHELL, n, ALPHA_AI, BETA_AI, fr, turns)
    print(f"    {label.replace(chr(10), ' '):35s}  Q_total={attr['total']*100:.0f}%"
          f"  (base={attr['baseline']*100:.0f}%"
          f"  topo=+{attr['topology']*100:.0f}%"
          f"  tao=+{attr['tao']*100:.0f}%"
          f"  verify=+{attr['verification']*100:.0f}%)")


print(f"\nDone. {'Plots displayed.' if SHOW else f'PNGs saved to {OUTPUT_DIR}/'}")
