import os
import pickle
import sys
import numpy as np
import random
from typing import List

# Ensure src directory is in path to import models and config
# This allows the script to find 'models.py' and 'config.py'
try:
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from models import Place
    import config
except ImportError:
    print("Error: Make sure models.py and config.py are in the 'src' directory.")
    sys.exit(1)

# --- Helper function for normalization ---

def _normalize_points(points: np.ndarray) -> np.ndarray:
    """Normalizes points to a standard 0-10 range."""
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    # Avoid division by zero if all points are identical
    range_vals = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
    normalized = (points - min_vals) / range_vals
    return normalized * 10

# --- Shape Generation Functions (No sklearn) ---

def create_bubble_data(K: int, random_state: int = 42) -> List[Place]:
    """
    Generates 'bubble' (blob) data using numpy.
    """
    print(f"  Generating 'bubble' with K={K}...")
    np.random.seed(random_state)
    
    # Define 5 cluster centers
    centers = np.array([
        [2, 2], [8, 2], [5, 5], [2, 8], [8, 8]
    ])
    num_centers = len(centers)
    
    # Assign K points proportionally to the centers
    points_per_center = [K // num_centers] * num_centers
    for i in range(K % num_centers): # Distribute the remainder
        points_per_center[i] += 1
        
    all_points = []
    for i in range(num_centers):
        num_points = points_per_center[i]
        # Generate points from a normal distribution around the center
        cluster_points = np.random.randn(num_points, 2) * 0.8 + centers[i]
        all_points.append(cluster_points)
        
    points = np.vstack(all_points)
    points = _normalize_points(points)
    
    # Convert to Place objects
    place_list = [Place(id=i, coords=(points[i, 0], points[i, 1])) for i in range(K)]
    return place_list

# --- *** COMPLETELY REWRITTEN FUNCTION *** ---
def create_s_curve_data(K: int, random_state: int = 42) -> List[Place]:
    """
    Generates 's_curve' as a figure-eight (infinity) shape
    to match the user's image.
    Uses a Lissajous curve with a 1:2 frequency ratio.
    """
    print(f"  Generating 's_curve' (infinity shape) with K={K}...")
    np.random.seed(random_state)
    
    # Generate K points evenly spaced from 0 to 2*pi
    t = np.linspace(0, 2 * np.pi, K)
    
    # Parametric equations for a figure-eight (Lissajous curve)
    x = np.sin(t) 
    y = np.sin(2 * t) # Key is the 1:2 ratio (t vs 2*t)
    
    # Combine into a 2D array
    points = np.vstack((x, y)).T
    
    # Add a *small* amount of Gaussian noise to scatter points
    # This ensures they are not "so close" but still form the shape.
    noise_factor = 0.1 # Reduced noise to make shape clearer
    points += np.random.randn(K, 2) * noise_factor
    
    points = _normalize_points(points)
    
    place_list = [Place(id=i, coords=(points[i, 0], points[i, 1])) for i in range(K)]
    return place_list
# --- *** END OF UPDATE *** ---

def create_flower_data(K: int, random_state: int = 42) -> List[Place]:
    """
    Generates 'flower' data using numpy (polar coordinates).
    """
    print(f"  Generating 'flower' with K={K}...")
    np.random.seed(random_state)
    
    num_petals = 6
    
    # Generate random angles (theta)
    t = np.random.uniform(0, 2 * np.pi, K)
    
    # Generate base radius using a sine function to create petals
    r_base = np.sin(t * num_petals / 2) ** 2
    
    # Add random noise to the radius
    r = r_base + np.random.uniform(0, 0.4, K)
    
    # Convert from polar (r, t) to cartesian (x, y)
    x = r * np.cos(t)
    y = r * np.sin(t)
    
    points = np.vstack((x, y)).T
    points = _normalize_points(points)
    
    place_list = [Place(id=i, coords=(points[i, 0], points[i, 1])) for i in range(K)]
    return place_list

# --- Main Script Logic ---

def generate_and_save_datasets(force_overwrite: bool = True):
    """
    Main function to generate and save all shape datasets
    based on K values from config.COMBO.
    
    :param force_overwrite: If True, will delete existing files first.
    """
    print("Starting dataset generation (no sklearn)...")
    
    # Define the shapes to generate
    shape_generators = {
        "bubble": create_bubble_data,
        "s_curve": create_s_curve_data, # This now points to the infinity shape
        "flower": create_flower_data,
    }
    
    # Get all unique K values from the config COMBO
    k_values = sorted(list(set([k_val for k_val, _ in config.COMBO])))
    
    # Get all shape names from the config
    shape_names = [name for name in config.DATASET_NAMES if name in shape_generators]
    
    if not k_values:
        print("No (K, k) combos found in config.py. Exiting.")
        return
        
    if not shape_names:
        print(f"No shape names (e.g., 'bubble', 's_curve', 'flower') found in config.DATASET_NAMES. Exiting.")
        return

    # Define the output directory (../datasets/)
    base_dir = os.path.dirname(__file__)
    datasets_dir = os.path.join(base_dir, "..", "datasets")
    
    # Create the directory if it doesn't exist
    os.makedirs(datasets_dir, exist_ok=True)
    print(f"Ensuring output directory exists: {datasets_dir}")
    
    # Loop over all shapes and K values to generate files
    for K in k_values:
        for shape_name in shape_names:
            
            output_filename = f"{shape_name}_K{K}.pkl"
            output_path = os.path.join(datasets_dir, output_filename)
            
            # --- NEW: Overwrite logic ---
            if os.path.exists(output_path) and force_overwrite:
                print(f"  Force overwrite: Deleting old file: {output_filename}")
                os.remove(output_path)
            elif os.path.exists(output_path) and not force_overwrite:
                print(f"Skipping, file already exists: {output_filename}")
                continue
            # --- End of Overwrite logic ---
                
            # Get the correct generator function
            generator_func = shape_generators[shape_name]
            
            # Create the data
            dataset = generator_func(K)
            
            # Save the data
            try:
                with open(output_path, "wb") as f:
                    pickle.dump(dataset, f)
                print(f"Successfully saved new: {output_filename}")
            except Exception as e:
                print(f"Error saving {output_filename}: {e}")
                
    print("\nDataset generation complete.")

if __name__ == "__main__":
    # Add a check to ensure config lists are not empty
    if not hasattr(config, "COMBO") or not config.COMBO:
        print("Error: config.py does not contain or has an empty 'COMBO' list.")
    elif not hasattr(config, "DATASET_NAMES") or not config.DATASET_NAMES:
        print("Error: config.py does not contain or has an empty 'DATASET_NAMES' list.")
    else:
        # This will now delete old .pkl files and create the new ones
        generate_and_save_datasets(force_overwrite=True)