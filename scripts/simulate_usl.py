"""
USL simulation and visualisation — docs/architecture/05-math-apparatus.md

Produces four plots saved to scripts/output/ (or shown interactively with --show):
  1. 01_usl_three_layers.png  — USL throughput curves for all three calibrated layers
  2. 02_cg_mean_effect.png    — Effect of CG_mean on N_max; how ADR corpus quality shifts the ceiling
  3. 03_pareto_matrix.png     — Topology Pareto matrix heatmap (7 topologies × 3 axes)
  4. 04_dark_knowledge_gap.png— J_eff distribution and task acceptance rate vs ADR corpus size

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
§ Definition 2  (USL formula) — usl()                        → Plots 1, 2
§ Definition 3  (CG)          — kappa_eff() uses CG_mean     → Plot 2
§ Definition 4  (κ_eff)       — kappa_eff(kappa_base, cg)    → Plots 1, 2
§ Definition 5  (Extended USL)— usl() + kappa_eff() compose  → Plot 2 right panel
§ Definition 9  (Pareto axes) — TOPOLOGIES (T, E, D tuples)  → Plot 3
§ Definition 10 (J_eff gate)  — J_EFF_GATE = 0.4            → Plot 4
§ Proposition 1 (N_max)       — n_max()                      → Plots 1, 2
§ Proposition 5 (frontier)    — TOPOLOGIES frontier flags    → Plot 3
§ §3 Calibration table        — LAYERS (must match doc table)→ Plots 1, 2

CONSTANTS THAT MUST STAY IN SYNC WITH THE DOC
----------------------------------------------
    J_EFF_GATE = 0.4   → docs/architecture/05-math-apparatus.md §4 (J_eff gate row)
    LAYERS             → §3 Calibration and §Proposition 1 calibrated ceilings table
                         (must also match CALIBRATION_TABLE in validate_math.py)
    TOPOLOGIES         → docs/guides/theory-to-implementation.md Pareto Summary table
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

def usl(N: np.ndarray, alpha: float, kappa: float) -> np.ndarray:
    return N / (1 + alpha * (N - 1) + kappa * N * (N - 1))


def n_max(alpha: float, kappa: float) -> float:
    return math.sqrt((1 - alpha) / kappa)


def kappa_eff(kappa_base: float, cg_mean: float) -> float:
    return kappa_base / cg_mean


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
    # (label, alpha, kappa_base, cg_mean, color)
    ("CPU cores  (α=0.02, κ_eff=0.0003, N_max≈57)",   0.02, 0.0003, 1.0, "#2563eb"),
    ("Human teams (α=0.10, κ_eff=0.0083, N_max≈10)",  0.10, 0.005,  0.6, "#16a34a"),
    ("AI agents  (α=0.15, κ_eff=0.025,  N_max≈6)",    0.15, 0.01,   0.4, "#dc2626"),
]

# § Definition 2, Definition 5, Proposition 1, §3 Calibration
# Visually confirms: X(1)=1, peaks at N_max, retrograde past peak, three-layer gap
print("\nPlot 1 — USL throughput curves")
fig, ax = plt.subplots(figsize=(10, 6))

N = np.linspace(1, 70, 500)

for label, alpha, kb, cg, color in LAYERS:
    ke = kappa_eff(kb, cg)
    X = usl(N, alpha, ke)
    nm = n_max(alpha, ke)
    X_peak = usl(np.array([nm]), alpha, ke)[0]

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
# Visually confirms: higher CG_mean → lower κ_eff → higher N_max (better ADR corpus = more agents)
print("Plot 2 — CG_mean vs N_max for AI-agent layer")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cg_values = np.linspace(0.2, 1.0, 200)
alpha_ai = 0.15
kb_ai = 0.01

n_max_values = [n_max(alpha_ai, kappa_eff(kb_ai, cg)) for cg in cg_values]

ax = axes[0]
ax.plot(cg_values, n_max_values, color="#7c3aed", linewidth=2)
ax.axhline(6,  color="#dc2626", linestyle="--", linewidth=1, label="CG_mean=0.4 (typical AI, N_max≈6)")
ax.axhline(10, color="#16a34a", linestyle="--", linewidth=1, label="CG_mean=0.6 (human team, N_max≈10)")
ax.axvline(0.4, color="#dc2626", linestyle=":", linewidth=0.8, alpha=0.6)
ax.axvline(0.6, color="#16a34a", linestyle=":", linewidth=0.8, alpha=0.6)
ax.set_xlabel("CG_mean (mean common ground across agent pairs)", fontsize=10)
ax.set_ylabel("N_max (scalability ceiling)", fontsize=10)
ax.set_title("Higher common ground → higher N_max\n"
             "Better ADR corpus and τ alignment raises the ceiling", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.2, 1.0)

# Right panel: USL curves at three CG_mean values
ax2 = axes[1]
N_short = np.linspace(1, 20, 200)
for cg, color, label in [
    (0.4, "#dc2626", "CG_mean=0.40 (typical AI)"),
    (0.6, "#f97316", "CG_mean=0.60 (with ADR corpus)"),
    (0.9, "#16a34a", "CG_mean=0.90 (high alignment)"),
]:
    ke = kappa_eff(kb_ai, cg)
    X = usl(N_short, alpha_ai, ke)
    nm = n_max(alpha_ai, ke)
    ax2.plot(N_short, X, color=color, linewidth=2, label=f"{label}  N_max≈{nm:.0f}")
    ax2.axvline(nm, color=color, linewidth=1, linestyle="--", alpha=0.4)

ax2.set_xlabel("Number of AI agents (N)", fontsize=10)
ax2.set_ylabel("Normalised throughput X(N)", fontsize=10)
ax2.set_title("USL curves at different CG_mean values\n"
              "ADR corpus quality directly shifts the scalability ceiling", fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, 20)

fig.suptitle("Effect of Common Ground on Scalability  (AI-agent layer, α=0.15, κ_base=0.01)",
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
# J_EFF_GATE must match docs/architecture/05-math-apparatus.md §4 and validate_math.py
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
ax.set_title("J_eff distribution vs ADR corpus size\n"
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
ax2.set_title("Task acceptance rate vs ADR corpus size\n"
              "0 ADRs → most tasks rejected; 5+ ADRs → near-full acceptance", fontsize=10)
ax2.set_ylim(0, 110)
ax2.set_xticks(adr_counts)
ax2.grid(True, axis="y", alpha=0.3)

fig.suptitle("Dark Knowledge Gap (J_eff) — Effect of ADR Corpus on Task Acceptance",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save_or_show(fig, "04_dark_knowledge_gap.png")

print(f"\nDone. {'Plots displayed.' if SHOW else f'PNGs saved to {OUTPUT_DIR}/'}")
