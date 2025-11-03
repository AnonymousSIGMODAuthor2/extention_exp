# raw_map_exp.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from typing import List, Dict
from models import Place
from config import COMBO, NUM_CELLS, DATASET_NAMES
from baseline_iadu import load_db_dataset
from hardcore_exp import plot_folium_map

EXPERIMENT_NAME = "raw_map_exp_allK"
SAVE_DIR = "raw_maps"

# collect all distinct Ks from COMBO
ALL_KS = sorted({K for (K, _k) in COMBO})
# pick a single grid just to drive the plot API; we don't care about it
ALL_GS = list(NUM_CELLS)
G_FIXED = 256 if 256 in ALL_GS else (ALL_GS[0] if ALL_GS else 256)
# dummy k to satisfy the plotting API/filename it creates before we rename
K_FIXED_FOR_API = 0  # we ignore k in our final filename

def move_map(shape: str, K: int, k_used: int, G_used: int):
    """Move/rename maps/{shape}_K{K}_k{k_used}_G{G_used}.html -> raw_maps/{shape}_K{K}.html"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    src = f"maps/{shape}_K{K}_k{k_used}_G{G_used}.html"
    dst = f"{SAVE_DIR}/{shape}_K{K}.html"
    if os.path.exists(src):
        if os.path.exists(dst):
            os.remove(dst)
        os.replace(src, dst)
        print(f"✔ Saved: {dst}")
    else:
        print(f"✗ Expected output not found at: {src}")

def run_experiment():
    print(f"[{EXPERIMENT_NAME}] G={G_FIXED}; ignoring k; iterating Ks={ALL_KS}")
    for shape in DATASET_NAMES:
        for K in ALL_KS:
            try:
                S: List[Place] = load_db_dataset(shape, K)
            except FileNotFoundError as e:
                print(f"  ✗ Missing dataset for shape={shape}, K={K} ({e})")
                continue

            # empty configs => plot only raw S via your API
            empty_configs: Dict[str, List[Place]] = {}
            # call your plotting function (it writes under maps/ first)
            plot_folium_map(shape, K, K_FIXED_FOR_API, G_FIXED, empty_configs, S)
            # move & rename into raw_maps without k/G in the name
            move_map(shape, K, K_FIXED_FOR_API, G_FIXED)

if __name__ == "__main__":
    run_experiment()
