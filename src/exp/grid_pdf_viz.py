# src/exp/grid_pdf_viz.py
import os
import sys
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

# ---- make 'src/' importable when running this file directly ----
_THIS_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ---- project imports (now that src/ is on sys.path) ----
from config import COMBO, NUM_CELLS, DATASET_NAMES
from baseline_iadu import load_db_dataset
from models import Place, Cell  # classes already in your repo

# Grid class may be named SquareGrid or SquereGrid depending on your file
try:
    from grid_iadu import SquareGrid as _GridClass
except ImportError:
    try:
        from grid_iadu import SquereGrid as _GridClass  # fallback alias
    except ImportError as e:
        raise ImportError(
            "Could not import SquareGrid/SquereGrid from grid_iadu.py. "
            "Open grid_iadu.py and export one of these names."
        ) from e


# ============================ Drawing ============================

def _get_bounds_and_cell(grid) -> Tuple[float, float, float, float, float]:
    """
    Extract (x_min, x_max, y_min, y_max, cell_size) from your grid class.
    Works with both SquareGrid and SquereGrid that store attributes like
    x_min/x_max/y_min/y_max and cell_w (or cell_h).
    """
    x_min = getattr(grid, "x_min")
    x_max = getattr(grid, "x_max")
    y_min = getattr(grid, "y_min")
    y_max = getattr(grid, "y_max")
    # prefer square cells if available; otherwise use cell_w
    cell_size = getattr(grid, "cell_w", None)
    if cell_size is None:
        raise AttributeError("Grid object has no attribute 'cell_w'. Add it or adapt this helper.")
    return x_min, x_max, y_min, y_max, float(cell_size)


def save_grid_pdf(
    grid,
    places: List[Place],
    out_pdf_path: str,
    *,
    title: str = "",
    show_cell_counts: bool = False,
    point_size: float = 6.0,
    grid_linewidth: float = 0.6,
    nonempty_alpha: float = 0.20,
) -> None:
    """Create a one-page PDF visualizing the grid, full cells, and point locations."""
    x_min, x_max, y_min, y_max, cell = _get_bounds_and_cell(grid)
    Ax = int(round((x_max - x_min) / cell))
    Ay = int(round((y_max - y_min) / cell))

    cells = grid.get_grid()                          # {(gx,gy): Cell}
    full_cells = [(idx, c) for idx, c in cells.items() if c.size() > 0]
    CL = len(full_cells)
    total_cells = Ax * Ay
    K = len(places)

    os.makedirs(os.path.dirname(out_pdf_path), exist_ok=True)
    with PdfPages(out_pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ttl = title or "Grid visualization"
        ax.set_title(f"{ttl}\nK={K} |G|={total_cells} |CL|={CL} avg_occ={K/max(1,CL):.2f}")

        # fill non-empty cells
        for (gx, gy), cell_obj in full_cells:
            x0 = x_min + gx * cell
            y0 = y_min + gy * cell
            ax.add_patch(Rectangle((x0, y0), cell, cell, facecolor="C0", alpha=nonempty_alpha, edgecolor="none"))
            if show_cell_counts:
                ax.text(x0 + cell / 2, y0 + cell / 2, str(cell_obj.size()), ha="center", va="center", fontsize=6)

        # grid lines
        for i in range(Ax + 1):
            x = x_min + i * cell
            ax.plot([x, x], [y_min, y_max], color="0.6", linewidth=grid_linewidth)
        for j in range(Ay + 1):
            y = y_min + j * cell
            ax.plot([x_min, x_max], [y, y], color="0.6", linewidth=grid_linewidth)

        # points
        pts = np.array([p.coords for p in places], dtype=float)
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], s=point_size, edgecolors="black", linewidths=0.2, facecolors="white", zorder=3)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"[OK] Saved grid PDF -> {out_pdf_path}  (K={K}, CL={CL}, G={total_cells})")


# ======================== Experiment runner ========================

EXPERIMENT_NAME = "grid_viz"
SHAPES = DATASET_NAMES

def run_grid_visualization(output_dir: str = "grid_pdfs") -> None:
    """
    Mirrors the structure of time experiments: for each shape, K, and G,
    load S, build a grid, and save a PDF snapshot.
    """
    os.makedirs(output_dir, exist_ok=True)

    for shape in SHAPES:
        for (K, _k_unused) in COMBO:
            # Load S for this shape and K
            try:
                S: List[Place] = load_db_dataset(shape, K)
            except Exception as e:
                print(f"[WARN] Skipping {shape} K={K}: {e}")
                continue

            for G in NUM_CELLS:
                try:
                    grid = _GridClass(S, G)
                except Exception as e:
                    print(f"[WARN] Grid failure for {shape} K={K} G={G}: {e}")
                    continue

                pdf_name = f"{shape}_K{K}_G{G}.pdf"
                out_path = os.path.join(output_dir, pdf_name)

                save_grid_pdf(
                    grid,
                    S,
                    out_pdf_path=out_path,
                    title=f"{shape} — K={K}, G={G}",
                    show_cell_counts=True,
                )

if __name__ == "__main__":
    run_grid_visualization()
