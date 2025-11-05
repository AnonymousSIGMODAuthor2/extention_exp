# config.py
import math


#NUM_CELLS = [64, 100, 256, 529, 1024]  # Different grid sizes for experiments

#default G = 256
NUM_CELLS = [25]


COMBO = [
    # (100, 20),
    # (200, 20),
    # (500, 20),
    # # # #(1000, 10),
    # # # #(1000, 15),
    (5000, 100),
    # #(1000, 50),
    # (5000, 20)
]

GAMMAS = [1]  # example values for g


DATASET_NAMES = [
    "1994_FIFA_World_Cup_squads",
]


SIMULATED_DATASETS = [
    # ex.: "bubbles",

]

# Generate GRID_RANGE dynamically based on NUM_CELLS
def get_grid_range_for_cells(num_cells: int, cell_size: float = 1.0) -> tuple:
    G = int(math.sqrt(num_cells))
    return (0, G * cell_size)

CELL_SIZE = 1.0
GRID_RANGES = {g: get_grid_range_for_cells(g, CELL_SIZE) for g in NUM_CELLS}
