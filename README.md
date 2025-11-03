# Top-k Representative Spatial Objects using Spatial Diversified Proportionality

---

## Algorithms

The main algorithms from the paper are implemented in the `src/` directory:

* **Algorithm 1: Pruning of $S$ (or Retrieval of $R$ ) using Biased Sampling**: Implemented in `src/biased_sampling.py` and used by the hybrid methods in `src/hybrid_sampling.py`.
* **Algorithm 2: Virtual Grid Based Algorithm**: Implemented as `virtual_grid_based_algorithm` in `src/grid_iadu.py` for approximating $S^S$ scores.
* **Algorithm 3: Baseline $IAdU$**: Implemented as `baseline_iadu_algorithm` in `src/baseline_iadu.py`. The exact $S^S$ pre-computation (`base_precompute`) is also in this file.
* **Algorithm 4: Grid based $IAdU$**: Implemented as `grid_based_iadu_algorithm` in `src/grid_iadu.py`, utilizing a heap-per-cell optimization.

---

## Running the Experiments

### 1. Dependencies

Install the required Python 3 libraries:

```bash
pip install numpy pandas scikit-learn matplotlib openpyxl folium
```


### 2. **Configuration**

All experiment parameters are set in src/config.py. You can edit this file to define the combinations of $K$ (initial set size) and $k$ (result set size), as well as the grid granularities ( $G$ ) to be tested.


### 3. **Datasets**

The experiments use queries derived from the datasets **DBpedia** and **YAGO2**, as described in the paper. The pre-processed queries are expected to be in the `datasets/` directory.

#### Generating Queries

Run the generator scripts (`src/dbpedia_query_generator.py` and `src/yago2_query_generator.py`) to process raw data (like `pid.txt`). These scripts create `.pkl` files containing queries (subsets of places) for different $K$ values.

Move the generated `.pkl` files from their output folders (`dbpedia_output/` and `yago_square/`) into the `datasets/` directory so the experiment scripts can find them.

* **DBpedia**: [https://www.dbpedia.org](https://www.dbpedia.org)
* **YAGO2**: [https://www.mpi-inf.mpg.de/departments/databases-andinformation-systems/research/yago-naga/yago/](https://www.mpi-inf.mpg.de/departments/databases-andinformation-systems/research/yago-naga/yago/)



### 4. **Running the Experiments**

The experiment scripts are located in the `src/exp/` directory. To run an experiment, you can execute the desired Python script from the `src/` directory.

For example, to run the `hardcore_exp.py` experiment, you would run the following command from the root of the project:

```bash
python src/exp/hardcore_exp.py
```

This will run the experiment with the parameters you have defined in `src/config.py`. The script will likely iterate through all the configured datasets, grid sizes, and other parameters, running the necessary algorithms and saving the results.



### 5. **Results**

The output of the experiments, such as logs, plots, or raw data, will be saved to a directory (e.g., `results/` or `output/`). You may need to create this directory if the scripts don't do it automatically.
