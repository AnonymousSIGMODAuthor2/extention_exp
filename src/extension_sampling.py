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


def grid_sampling(S: List[Place], k: int, W: float, G: int):
    """
    Implements "Option 3: Grid based and proportional selection per cell".
    
    This method allocates k samples proportionally to grid cells based on their
    density, then performs simple random sampling within each cell.
    
    NOW ALSO RETURNS:
    - cell_stats: Dict[tuple, Tuple[int, int]] mapping cell_id -> (total, selected)
    """
    
    # --- Preparation step (STEP 1) ---
    # This step's purpose *is* to create the grid and CL.
    # Allocation is inherent to the operation.
    t_prep_start = time.time()
    grid = SquareGrid(S, G)
    CL = grid.get_full_cells() # Get non-empty cells
    prep_time = time.time() - t_prep_start
    
    optimal_psS, optimal_sS, optimal_prep_time = base_precompute(S)
    
    K = len(S)
    if K == 0 or k == 0:
        return [], 0.0, 0.0, 0.0, prep_time, 0.0, 0, {}
    # ---

    # --- Selection step (STEP 2) ---
    
    # (!!) DECLARE all data structures *BEFORE* timing (!!)
    R: List[Place] = []
    k_alloc: Dict[Tuple[int, int], int] = {} 
    cell_stats: Dict[Tuple[int, int], Tuple[int, int]] = {}
    remainders = [] 
    total_k_allocated = 0
    # Create the cell_map lookup *before* timing as well
    cell_map = {c.id: c for c in CL} 

    # (!!) START TIMER *AFTER* declarations (!!)
    t_selection_start = time.time()
    
    # --- Proportionality allocation ---
    # This loop is now just computation + populating existing dicts/lists
    for c in CL:
        if c.size() == 0:
            continue
        
        ideal = k * (c.size() / K)
        integer_part = math.floor(ideal)
        
        k_alloc[c.id] = integer_part
        total_k_allocated += integer_part
        remainders.append((c.id, ideal - integer_part)) # Appending is a computation

    k_remaining = k - total_k_allocated
    
    # Sorting is a computation
    remainders.sort(key=lambda x: x[1], reverse=True)
    
    # This loop is computation
    for i in range(min(k_remaining, len(remainders))):
        c_id_to_add = remainders[i][0]
        
        # (!!) --- THIS IS THE FIX --- (!!)
        # Changed "c.id_to_add" to "c_id_to_add"
        k_alloc[c_id_to_add] += 1
        
    # --- Random selection ---
    # This loop is computation (lookup, sampling, extending list)
    for cell_id, num_to_pick in k_alloc.items():
        if num_to_pick > 0:
            cell = cell_map[cell_id]
            actual_pick = min(num_to_pick, cell.size())
            
            if actual_pick > 0:
                # .extend and random.sample are the core computations here
                R.extend(random.sample(cell.places, actual_pick))

    # (!!) STOP TIMER (!!)
    selection_time = time.time() - t_selection_start
    
    # --- NEW: Build cell_stats dictionary ---
    # maps cell_id -> (total_count, selected_count)
    for c in CL: # Use all non-empty cells
        cell_id = c.id
        total_count = c.size()
        selected_count = k_alloc.get(cell_id, 0) # Get from k_alloc
        cell_stats[cell_id] = (total_count, selected_count)
    # ---

    # --- Compute final scores (Using HPFR as requested) ---
    if not R:
        # Throw Exception
        raise ValueError("Selected sample R is empty, cannot compute HPFR.")
        
    score, sum_psS, sum_psR = HPFR(R, optimal_psS, optimal_sS, W, K)
    
    # --- MODIFIED RETURN (8 values) ---
    return R, score, sum_psS, sum_psR, prep_time, selection_time, len(CL), cell_stats