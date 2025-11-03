# plot_yago_queries.py — minimal YAGO map using only load_yago_dataset
from __future__ import annotations
import os, sys
from typing import List, Tuple
import folium
from folium.plugins import MarkerCluster

# ---- derive paths ----
HERE = os.path.abspath(os.path.dirname(__file__))           # .../src/exp
SRC_ROOT = os.path.abspath(os.path.join(HERE, ".."))        # .../src
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_ROOT, ".."))# project root (where yago_datasets lives)

# make src importable
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# *** make relative paths (used by your loader) resolve from project root ***
os.chdir(PROJECT_ROOT)

from config import YAGO_DATASET_NAMES, COMBO
from baseline_iadu import load_yago_dataset  # <- your loader; expects 'yago_datasets/..' relative to CWD

# ---- helpers ----
def as_tuples(data) -> List[Tuple[int, float, float]]:
    """Normalize to list of (id, x=lon, y=lat)."""
    if not data:
        return []
    first = data[0]
    if hasattr(first, "coords"):  # Place-like
        out = []
        for p in data:
            pid = int(getattr(p, "id", 0))
            x, y = p.coords  # (lon, lat)
            out.append((pid, float(x), float(y)))
        return out
    # already tuples
    return [(int(pid), float(x), float(y)) for pid, x, y in data]

def bbox_xy(pts: List[Tuple[int, float, float]]):
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)

def color_for_k(k: int) -> str:
    return {100: "#1f77b4", 200: "#2ca02c", 500: "#9467bd",
            1000: "#ff7f0e", 2000: "#d62728"}.get(k, "#7f7f7f")

def ks_from_combo(combo_list):
    """Extract K from entries like (K, 20) or just K."""
    out = []
    for c in combo_list:
        out.append(int(c[0]) if isinstance(c, (list, tuple)) else int(c))
    return sorted(set(out))

# ---- main ----
def main():
    Ks = ks_from_combo(COMBO)

    # preload for global center
    all_pts: List[Tuple[int, float, float]] = []
    layers: List[Tuple[str, int, List[Tuple[int, float, float]]]] = []

    for name in YAGO_DATASET_NAMES:
        for K in Ks:
            data = load_yago_dataset(name, K)   # <- works now because CWD = project root
            pts = as_tuples(data)
            if not pts:
                print(f"[WARN] empty: {name} K={K}")
                continue
            layers.append((name, K, pts))
            all_pts.extend(pts)

    if not all_pts:
        raise RuntimeError("No points loaded. Check COMBO, config names, and dataset files.")

    xmin = min(p[1] for p in all_pts); xmax = max(p[1] for p in all_pts)
    ymin = min(p[2] for p in all_pts); ymax = max(p[2] for p in all_pts)
    center = [(ymin + ymax) / 2.0, (xmin + xmax) / 2.0]

    m = folium.Map(location=center, zoom_start=9, tiles="CartoDB positron", control_scale=True)

    # a layer per (query, K)
    for name, K, pts in layers:
        layer_name = f"{name} · K={K}"
        fg = folium.FeatureGroup(name=layer_name, show=False)
        mc = MarkerCluster(name=layer_name)
        fg.add_child(mc)

        col = color_for_k(K)
        for pid, x, y in pts:  # (id, lon, lat)
            folium.CircleMarker(
                location=[y, x],
                radius=2,
                color=col,
                weight=0,
                fill=True,
                fill_opacity=0.75,
                tooltip=f"{name} | K={K} | id={pid}",
            ).add_to(mc)

        # quick bbox outline
        xmin, ymin, xmax, ymax = bbox_xy(pts)
        folium.Rectangle([[ymin, xmin], [ymax, xmax]], color=col, weight=1, fill=False, opacity=0.6).add_to(fg)

        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # save under src/exp/maps_out (create if needed)
    out_dir = os.path.join(SRC_ROOT, "exp", "maps_out")
    os.makedirs(out_dir, exist_ok=True)
    out_html = os.path.join(out_dir, "yago_queries_map.html")
    m.save(out_html)
    print(f"[OK] Map saved → {out_html}")

if __name__ == "__main__":
    main()
