# IA26_P01_G03 - Local Food Supply Chain Optimization

Artificial Intelligence Project 01 (2026–2027) - **Group 03**

## The Problem

A routing agent decides which vehicle collects each batch of products from local producers and the order in which pickups and deliveries are performed.

The agent must respect time windows, vehicle capacity constraints, mandatory warehouse visits (maximum 24 hours between collection and warehouse delivery), while minimizing the total distance traveled.

| Algorithm | File | Purpose |
|---|---|---|
| Greedy Best-First Search and A* (2 heuristics) | `src/informed.py` | Determine the stop order for each vehicle |
| Hill Climbing (steepest ascent / first improvement) | `src/local_search.py` | Improve the Greedy solution |
| CSP (backtracking, forward checking, AC-3) | `src/csp.py` | Assign pickup batches to vehicles |
| Model, cost function and route validator | `src/model.py` | Common foundation for all algorithms |

## Project Structure

```text
IA26_P01_G03/
├── data_tiny/                     # Tiny dataset CSV files
├── data_small/                    # Small dataset CSV files
├── src/
│   ├── model.py                   # Entities, cost function and route validator
│   ├── informed.py                # Greedy, A* and heuristics
│   ├── local_search.py            # Hill Climbing
│   └── csp.py                     # Backtracking, forward checking and AC-3
├── Notebooks/
│   ├── 01_Project01_G03.ipynb     # Main project notebook
│   └── 02_eda_tiny.ipynb          # Exploratory data analysis
├── readme.txt
└── requirements.txt
```

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch the main Jupyter Notebook:

```bash
jupyter notebook Notebooks/01_Project01_G03.ipynb
```

## Main Results

Total distance traveled (in kilometers):

| Dataset | Greedy | A* | Greedy + Hill Climbing |
|---|---:|---:|---:|
| Tiny | 660.61 km | **580.90 km** | **580.90 km** |
| Small | 1159.31 km | **1129.32 km** | **1129.32 km** |