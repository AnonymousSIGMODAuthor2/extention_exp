import copy
import pandas as pd
from typing import List, Tuple, Dict
from math import floor
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from HPF_eq import HPFR, HPFR_div
from models import Place
import config as cfg



# --- subfunction ---
def baseline_iadu_algorithm(S: List[Place], K_full: int, k: int, W: float, psS, sS) -> Tuple[List[Place], float]:
    R = []
    K = K_full
    candidates = copy.deepcopy(S)

    for p in candidates:
        #+ p.rF
        p.cHPF = psS[p.id] + p.rF
        #p.cHPF = 0
    
    select_start = time.time()
    while len(R) < k:
        curMP = max(candidates, key=lambda p: p.cHPF)
        candidates.remove(curMP)
        R.append(curMP)
        if len(R) < k:
            for p in candidates:
                if p.id != curMP.id:
                    p.cHPF += (K - k) * (p.rF + curMP.rF) / (k - 1) + (psS[p.id] + psS[curMP.id]) / (k - 1) - 2  * W * spacial_proximity(sS, p, curMP)
    select_end = time.time()
    
    select_end - select_start
    return R, select_end - select_start

############################################################################################################
# use symmetric sS
def spacial_proximity(sS, pi, pj):
    return sS.get((pi.id, pj.id)) or sS.get((pj.id, pi.id)) or 0.0


def base_precompute(S: List[Place]) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
    sS = {}
    maxD = maxDistance(S)
    # Initialize psS
    psS = {p.id: 0.0 for p in S}
    
    prep_start = time.time()
    for i in range(len(S)):
        pi = S[i]
        for j in range(i + 1, len(S)):
            pj = S[j]
            sim = 0
            if pi.id != pj.id:
                # Calculate Euclidean distance and similarity
                d = np.linalg.norm(pi.coords - pj.coords)
                sim = 1 - d / maxD
                
                # Store similarity in the dictionary
                sS[pi.id, pj.id] = sim
                # Update psS
                psS[pj.id] += sim
                psS[pi.id] += sim
    prep_end = time.time()
            
    return psS, sS, prep_end - prep_start

####################################################################################################
#####################################################################################################
# --- IAdU method ---
def iadu(S: List[Place], k: int, W) -> Tuple[List[Place], Dict[int, float], Dict[int, float], float, float, float]:
    K = len(S)
    # Preparation step
    exact_psS, exact_sS, prep_time = base_precompute(S)
        
    # Run baseline IAdU algorithm
    R, selection_time = baseline_iadu_algorithm(S, K, k, W, exact_psS, exact_sS)
    
    # Compute final scores
    score, sum_psS, sum_psR = HPFR(R, exact_psS, exact_sS, W, len(S))
    
    return R, score, sum_psS, sum_psR, prep_time, selection_time

# --- IAdU method ---
def iadu_div(S: List[Place], k: int, W) -> Tuple[List[Place], Dict[int, float], Dict[int, float], float, float, float]:
    K = len(S)
    # Preparation step
    exact_psS, exact_sS, prep_time = base_precompute(S)
        
    # Run baseline IAdU algorithm
    R, selection_time = baseline_iadu_algorithm(S, K, k, W, exact_psS, exact_sS)
    
    # Compute final scores
    score_rf, score_ps, sum_psS, sum_psR = HPFR_div(R, exact_psS, exact_sS, W, len(S))
    
    return R, score_rf + score_ps, score_rf, score_ps, sum_psS, sum_psR, prep_time, selection_time

# import pickle

# def load_dataset(name: str, K: int):
#     path = f"datasets/{name}_K{K}.pkl"
#     with open(path, "rb") as f:
#         return pickle.load(f)

def load_db_dataset(region_name: str, K: int) -> List[Place]:
    """
    Load a DBpedia subregion dataset (exact K places) from db_datasets folder.
    """
    import pickle, os
    from models import Place
    fname = f"dbpedia_{region_name}_K{K}.pkl"
    path = os.path.join("db_datasets", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No dataset file found: {path}")
    with open(path, "rb") as f:
        data: List[Place] = pickle.load(f)
    return data

def load_yago_dataset(region_name: str, K: int) -> List[Place]:
    """
    Load a DBpedia subregion dataset (exact K places) from db_datasets folder.
    """
    import pickle, os
    from models import Place
    fname = f"yago_{region_name}_K{K}.pkl"
    path = os.path.join("yago_datasets", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No dataset file found: {path}")
    with open(path, "rb") as f:
        data: List[Place] = pickle.load(f)
    return data
from typing import List
import os, pickle, re
from models import Place

def load_dataset(shape: str, K: int, datasets_dir: str = None) -> List[Place]:
    """
    Load a dataset list[Place] for a given Wikipedia/YAGO title `shape` and cardinality K.
    Auto-detects YAGO vs DBpedia by filename prefix.

    Expected filenames inside datasets_dir:
        - dbpedia_<TITLE>_K{K}.pkl
        - yago_<TITLE>_K{K}.pkl

    The function is tolerant to minor punctuation/Unicode differences in <TITLE>.
    """
    # --- Locate datasets directory (../datasets preferred; fall back to ./datasets)
    here = os.path.abspath(os.path.dirname(__file__))
    candidates_dirs = []
    if datasets_dir is not None:
        candidates_dirs.append(os.path.abspath(datasets_dir))
    candidates_dirs.append(os.path.abspath(os.path.join(here, "..", "datasets")))
    candidates_dirs.append(os.path.abspath(os.path.join(here, "datasets")))
    base_dir = next((d for d in candidates_dirs if os.path.isdir(d)), candidates_dirs[0])

    # --- Candidate exact paths (fast path)
    exact_candidates = [
        os.path.join(base_dir, f"yago_{shape}_K{K}.pkl"),
        os.path.join(base_dir, f"dbpedia_{shape}_K{K}.pkl"),
    ]
    for path in exact_candidates:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)

    # --- Fallback: tolerant scan (handles ’ vs ', en dash vs hyphen, etc.)
    def norm(s: str) -> str:
        # unify dashes/apostrophes, keep alnum/underscore, collapse repeats
        s = s.replace("–", "-").replace("—", "-").replace("’", "'")
        s = s.replace("´", "'").replace("`", "'")
        # keep letters/numbers/underscore only
        s = re.sub(r"[^0-9A-Za-z_'-]+", "_", s)
        # treat apostrophes like underscore for filename matching
        s = s.replace("'", "_").replace("-", "_")
        s = re.sub(r"_+", "_", s).strip("_").lower()
        return s

    target = norm(shape)
    best_path = None
    for fname in os.listdir(base_dir):
        if not fname.endswith(f"_K{K}.pkl"):
            continue
        if not (fname.startswith("dbpedia_") or fname.startswith("yago_")):
            continue
        # strip prefix & suffix to compare the title part
        title_part = fname.split("_K")[0]
        title_part = title_part.split("dbpedia_", 1)[-1] if title_part.startswith("dbpedia_") else title_part.split("yago_", 1)[-1]
        if norm(title_part) == target:
            best_path = os.path.join(base_dir, fname)
            break

    if best_path is None:
        # last-resort: loosen to "starts with" (helps when your stored title had extra tail)
        for fname in os.listdir(base_dir):
            if not fname.endswith(f"_K{K}.pkl"):
                continue
            if not (fname.startswith("dbpedia_") or fname.startswith("yago_")):
                continue
            title_part = fname.split("_K")[0]
            title_part = title_part.split("dbpedia_", 1)[-1] if title_part.startswith("dbpedia_") else title_part.split("yago_", 1)[-1]
            if norm(title_part).startswith(target):
                best_path = os.path.join(base_dir, fname)
                break

    if best_path and os.path.exists(best_path):
        with open(best_path, "rb") as f:
            return pickle.load(f)

    # If we reach here, nothing matched
    searched = "\n  - " + "\n  - ".join(exact_candidates)
    raise FileNotFoundError(
        f"[load_dataset] Could not find dataset for shape='{shape}', K={K} in '{base_dir}'. "
        f"Tried:{searched}\nAlso scanned folder with tolerant matching."
    )

def maxDistance(S: List[Place]) -> float:
    maxD = 0
    coords = np.array([p.coords for p in S])
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            d = np.linalg.norm(coords[i] - coords[j])
            maxD = max(maxD, d)
    return maxD

# --- Plot utility ---
def plot_selected(S: List[Place], R: List[Place], title: str, ax):
    coords = np.array([p.coords for p in S])
    ax.scatter(coords[:, 0], coords[:, 1], c="lightblue", s=10, label="All Places")

    # Extract coords by affinity value
    selected_1 = [p.coords for p in R if p.rF == 0.0]
    selected_08 = [p.coords for p in R if p.rF == 0.8]
    selected_06 = [p.coords for p in R if p.rF == 0.6]
    selected_04 = [p.coords for p in R if p.rF == 0.4]

    if selected_1:
        selected_1 = np.array(selected_1)
        ax.scatter(selected_1[:, 0], selected_1[:, 1], c="red", s=25, label="rF=0.0")
    if selected_08:
        selected_08 = np.array(selected_08)
        ax.scatter(selected_08[:, 0], selected_08[:, 1], c="black", s=25, label="rF=0.8")
    if selected_06:
        selected_06 = np.array(selected_06)
        ax.scatter(selected_06[:, 0], selected_06[:, 1], c="purple", s=25, label="rF=0.6")
    if selected_04:
        selected_04 = np.array(selected_04)
        ax.scatter(selected_04[:, 0], selected_04[:, 1], c="blue", s=25, label="rF=0.4")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")

baseline_scores = []
baseline_prep_times = []
iadu_scores = []

all_K_k = cfg.COMBO


