import sys
import os
import time
import math
import random
from typing import List, Dict, Tuple

# Ensure parent directory is in path to import models, HPF_eq, etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from grid_iadu import virtual_grid_based_algorithm
from models import Place
from models import SquareGrid  # Assuming SquareGrid is in models.py
# Using HPFR as you requested
from HPF_eq import HPFR 
from baseline_iadu import base_precompute


def grid_proportional_sampling(S: List[Place], k: int, W: float, G: int):
    """
    Implements "Option 3: Grid based and proportional selection per cell" [cite: 32-34].
    
    This method allocates k samples proportionally to grid cells based on their
    density, then performs simple random sampling within each cell.
    """
    
    # --- Preparation step (as in your file) ---
    t_prep_start = time.time()
    try:
        # 1. "allocate each object in a cell" 
        grid = SquareGrid(S, G)
    except ValueError:
        print("Warning: S is empty or invalid, returning empty results.")
        return [], 0.0, 0.0, 0.0, 0.0, 0.0, 0
        
    # 2. Get the "virtual grid" (list of non-empty cells)
    CL = grid.get_full_cells() 
    
    # 3. Calculate scores using VGA (as you said)
    #    Your template times this as prep_time.
    psS, sS , _ = virtual_grid_based_algorithm(CL,S)
    prep_time = time.time() - t_prep_start
    
    # 4. Get optimal scores for final H(R) calculation (as in your file)
    optimal_psS, optimal_sS, optimal_prep_time = base_precompute(S)
    
    K = len(S)
    if K == 0 or k == 0:
        return [], 0.0, 0.0, 0.0, prep_time, 0.0, 0
    # ---

    # --- Selection step ---
    # This is: "Then we pick from each cell proportionally objects" 
    # This uses the 'CL' (virtual grid) from the prep step.
    t_selection_start = time.time()
    
    R: List[Place] = []
    k_alloc: Dict[Tuple[int, int], int] = {} 
    remainders = [] 
    total_k_allocated = 0
    
    # 1. Calculate ideal picks, integer parts, and remainders
    for c in CL:
        if c.size() == 0:
            continue
        
        ideal = k * (c.size() / K)
        integer_part = math.floor(ideal)
        
        k_alloc[c.id] = integer_part
        total_k_allocated += integer_part
        remainders.append((c.id, ideal - integer_part))

    # 2. Distribute remaining k's based on largest remainders
    k_remaining = k - total_k_allocated
    
    remainders.sort(key=lambda x: x[1], reverse=True)
    
    for i in range(min(k_remaining, len(remainders))):
        c_id_to_add = remainders[i][0]
        k_alloc[c_id_to_add] += 1
        
    # 3. Build R by sampling from each cell
    cell_map = {c.id: c for c in CL} 

    for cell_id, num_to_pick in k_alloc.items():
        if num_to_pick > 0:
            cell = cell_map[cell_id]
            actual_pick = min(num_to_pick, cell.size())
            
            if actual_pick > 0:
                R.extend(random.sample(cell.places, actual_pick))

    selection_time = time.time() - t_selection_start
    # --- End of Selection Step ---
    
    
    # --- Compute final scores (Using HPFR as requested) ---
    if not R:
        return [], 0.0, 0.0, 0.0, prep_time, selection_time, len(CL)
        
    # Calling HPFR, using optimal scores as in your template
    score, sum_psS, sum_psR = HPFR(R, optimal_psS, optimal_sS, W, K)
    
    # --- CORRECTED RETURN ---
    return R, score, sum_psS, sum_psR, prep_time, selection_time, len(CL)