# Time-Decayed Caching (TDC)

A novel cache replacement policy that unifies recency and frequency signals through exponential time decay. This repository contains the implementation, the full experimental evaluation framework, and the resulting academic paper.

## Overview

Traditional cache replacement policies treat recency and frequency as separate signals. LRU evicts the least recently used item, LFU evicts the least frequently used. Each performs well on specific workloads but poorly on others. LFU, for instance, fails catastrophically when the popularity distribution shifts over time.

**Time-Decayed Caching (TDC)** addresses this by assigning each cached item a score defined as:

$$\text{Score}(x, t) = \sum_{i \in \text{accesses}(x)} e^{-\lambda\,(t - t_i)}$$

where $\lambda > 0$ is a decay rate parameter and $t_i$ is the timestamp of each past access to item $x$. When the cache is full, the item with the lowest score is evicted.

## Repository Structure

```
time-decayed-caching/
  code/
    cache_base.py               # Abstract base class for all cache policies
    cache_tdc.py                # TDC - proposed policy
    cache_lru.py                # LRU baseline
    cache_lfu.py                # LFU baseline
    cache_arc.py                # ARC baseline
    trace_generator.py          # Synthetic trace generation (Zipf, non-stationary phases)
    experiment.py               # Experiment runner and delta metric computation
    visualization.py            # Publication-ready plots and tables
    main.py                     # Run all synthetic trace experiments
    run_realworld_experiments.py # Run NASA HTTP log experiments
  data/
    NASA_access_log_Jul95.gz    # NASA HTTP access log (July 1995)
  paper/                        # Research paper files
```

## Getting Started

### Prerequisites

- Python 3.8+
- NumPy
- Pandas
- Matplotlib

### Installation

```bash
git clone <repository-url>
cd time-decayed-caching/code
pip install numpy pandas matplotlib
```

### Running Experiments

**Synthetic traces** (stationary Zipf + non-stationary phase changes):

```bash
python main.py
```

Outputs:
- `delta_table.png`: summary table of hit ratio deltas across all cache sizes
- `sliding_window_graph.png`: sliding window hit ratio over time for non-stationary traces

**Real-world trace** (NASA HTTP log):

```bash
python run_realworld_experiments.py
```

Requires `data/NASA_access_log_Jul95.gz` to be extracted as `data/access_log_Jul95`.

Outputs:
- `delta_table_realworld.png`: summary table for the NASA trace
- `sliding_window_realworld.png`: sliding window hit ratio over time

## Additional

This project is part of academic research. See the paper directory.
