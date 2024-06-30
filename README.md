# Codebase of Sequential Stackelberg Drone Defense

## Description

This project includes several Python scripts and modules for working with graphs, algorithms, and data visualization.

## Directory Structure

- `algorithms/`: Contains Python scripts for various algorithms including graph coarsening, a greedy baseline, and single and multi-drone strategies.
- `data/`: Directory for storing the raw city graph data and reward assigned cities.
- `results/`: Directory for storing result files.

## Key Files

- `export_graphs_with_rewards.py`: Script for exporting graphs with rewards.
- `s2d2_compute_multi_drone_sol.py`: Script for computing multi-drone solutions.
- `cross_compare_methods.py`: Script for cross-running the algorithm, e.g., baseline attacker vs. S2D2 Defender.
- `utils.py`: Contains utility functions used across the project.
- `visualize_results.py`: Script for visualizing results.

## Setup

To set up the environment for this project, use the provided `strategy_env.yml` file.

```sh
conda env create -f strategy_env.yml
```

NOTE: You will need Gurobi installed in the machine with license activated.

Download the `S2D2_data.zip` file from here: https://drive.proton.me/urls/7A3JN1D1TG#Ss8Pi2a8apbt

Pleace `S2D2_data.zip` in the home directory and unzip. This will create the `data/` directory as explained earlier.

## Usage
Here are the sequence of code execution required to run the codebase

Step 1: Setup what cities you want to do experiments with. Inside ``utils.py``, update the ``['City 1', 'City 2']`` inside the ``get_test_city_list()`` with the number of cities you prefer to run the code with.

Step 2: Setup a config in the ``config.yaml`` from ``config_list.yaml`` file according to your preference

Step 3: Run reward assignment code as following:
```sh
python export_graphs_with_rewards.py
```
It will assign lognormal and zipf rewards to the cities you listed and save them under ``data/cities_lognormal_dist`` and ``data/cities_zipf_dist`` respectively

Step 3: Compute the S2D2 multi drone solution. This will save the results in the results folder
```sh
python s2d2_compute_multi_drone_sol.py
```

Step 4: Compute the final result by running cross comparison between the baseline and S2D2
```sh
python cross_compare_methods.py
```

Step 5: Finally, visualize the results by running
```sh
python visualize_results.py
```

Thanks!
