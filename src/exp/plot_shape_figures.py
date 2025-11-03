# plot_exp.py  — one PDF; each page = one (K,k) with all shapes

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import defaultdict
from typing import List
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from models import Place
from config import COMBO, NUM_CELLS  # NUM_CELLS kept if you still need timings
from baseline_iadu import load_dataset, iadu, plot_selected
from hybrid_sampling import hybrid
from grid_iadu import grid_iadu  # optional
from biased_sampling import biased_sampling
import numpy as np

import config as cfg

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.style']  = 'italic'   # makes all text italic
# optional:
mpl.rcParams['legend.fontsize'] = 12



SHAPE_NAMES = ["bubble", "s_curve"]   # use your existing datasets
METHODS = ["IAdU", "Hybrid", "Sampling"]        # columns order

def plot_selected_bw(S, R, title: str, ax=None, S_sampled=None):
    """
    Legend rules (built from real scatter handles, no proxies):
      - IAdU / Sampling:   entries ->  S-R , R
      - Hybrid:            entries ->  S-S', S'-R , R
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    import numpy as np

    hS = hSp = hR = None  # scatter handles

    # --- S (all places) ---
    coords = np.array([p.coords for p in S], dtype=float) if S else np.zeros((0, 2))
    if len(coords):
        if S_sampled:  # HYBRID page
            hS = ax.scatter(coords[:, 0], coords[:, 1],
                            s=8, c="#87CEFA", alpha=0.35, edgecolors="none", label="S-S'")
        else:          # IAdU / SAMPLING page
            hS = ax.scatter(coords[:, 0], coords[:, 1],
                            s=8, c="#87CEFA", alpha=0.35, edgecolors="none", label="S-R")

    # --- S' (only for Hybrid) ---
    if S_sampled:
        ss = np.array([p.coords for p in S_sampled], dtype=float)
        if len(ss):
            hSp = ax.scatter(ss[:, 0], ss[:, 1],
                             s=14, c="red", alpha=0.90, edgecolors="none", label="S'-R")

    # --- R (selected) ---
    if R:
        if isinstance(R, tuple):  # robustness
            R = R[0]
        sel = np.array([p.coords for p in R], dtype=float)
        if len(sel):
            hR = ax.scatter(sel[:, 0], sel[:, 1],
                            s=29, c="black", alpha=0.95, edgecolors="none", label="R")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(True, alpha=0.3); ax.axis("equal")

    # ---- Legend (no box), built from real handles; ordered as you want ----
    handles = [h for h in (hS, hSp, hR) if h is not None]
    labels  = [h.get_label() for h in handles]
    leg = ax.legend(
        handles, labels,
        loc="upper right", bbox_to_anchor=(0.965, 0.985),
        frameon=False,
        handlelength=0,          # dot only
        scatterpoints=1, markerscale=1.6,
        handletextpad=0.65,      # <-- more gap so dots don’t touch letters
        labelspacing=0.20,
        borderaxespad=0.12,
        fontsize=13,
    )



# --- update the compute_* wrappers so they ALL return (R, S_sampled_or_None) ---
def compute_iadu(S, K, k, W):
    R_base, *_ = iadu(S, k, W)
    return R_base, None

def compute_hybrid(S, K, k, W):
    K_sample = int(getattr(cfg, "K_SAMPLE_RATIO", 0.20) * K) or max(1, K // 10)
    R_hybrid, S_sampled, *_ = hybrid(S, k, K_sample, W)
    return R_hybrid, S_sampled

def compute_sampling(S, K, k, W):
    R_biased, *_ = biased_sampling(S, k, W)
    return R_biased, None

METHOD_FUN = {
    "IAdU": compute_iadu,
    "Hybrid": compute_hybrid,
    "Sampling": compute_sampling,
}

# --- replace your run_experiment() with this ---
def run_experiment():
    pdf_path = f"figures.pdf"
    with PdfPages(pdf_path) as pdf:
        for (K, k) in COMBO:                         # from config
            W = 0.5 * K / k
            nrows, ncols = len(SHAPE_NAMES), len(METHODS)
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(5.0 * ncols, 4.5 * nrows),
                                     squeeze=False)

            for i, shape in enumerate(SHAPE_NAMES):
                try:
                    S = load_dataset(shape, K)       # uses "{shape}_K{K}.pkl"
                except FileNotFoundError:
                    # mark missing dataset, keep layout consistent
                    for j in range(ncols):
                        ax = axes[i, j]
                        ax.text(0.5, 0.5, f"Missing dataset:\n{shape}_K{K}.pkl",
                                ha="center", va="center")
                        ax.set_axis_off()
                    continue

                # --- inside run_experiment(), change the plotting call loop to this ---
                for j, method in enumerate(METHODS):
                    ax = axes[i, j]
                    R, S_samp = METHOD_FUN[method](S, K, k, W)
                    title = f"{shape} — {method}  (K={K}, k={k})"
                    plot_selected_bw(S, R, title, ax=ax, S_sampled=S_samp)

            fig.suptitle(f"PSS + IAdU — K={K}, k={k}", y=0.995, fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved multi-page PDF: {pdf_path}")

if __name__ == "__main__":
    run_experiment()
