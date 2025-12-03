"""
Experiment runner for cache evaluation.
"""

import numpy as np
from typing import Dict, List

from cache_tdc import TimeDecayedCache
from cache_lru import LRUCache
from cache_lfu import LFUCache
from cache_arc import ARCCache


class ExperimentRunner:
    """Run caching experiments and collect metrics."""
    
    def __init__(self):
        self.algorithms = {
            'Proposed': lambda c: TimeDecayedCache(c, decay_rate=0.995),
            'LRU': lambda c: LRUCache(c),
            'LFU': lambda c: LFUCache(c),
            'ARC': lambda c: ARCCache(c)
        }
    
    def run_single_experiment(self, trace: np.ndarray, cache_size: int, 
                               algorithm_name: str) -> Dict:
        """Run a single experiment with one algorithm on one trace."""
        cache = self.algorithms[algorithm_name](cache_size)
        
        for item in trace:
            cache.access(int(item))
        
        return cache.get_stats()
    
    def run_sliding_window_experiment(self, trace: np.ndarray, cache_size: int,
                                       window_size: int = 1000) -> Dict[str, List[float]]:
        """
        Run experiment tracking sliding window hit ratios.
        
        Returns hit ratio over time for each algorithm.
        """
        results = {name: [] for name in self.algorithms}
        caches = {name: self.algorithms[name](cache_size) for name in self.algorithms}
        
        # Track hits in sliding window
        window_hits = {name: [] for name in self.algorithms}
        
        for i, item in enumerate(trace):
            for name, cache in caches.items():
                hit = cache.access(int(item))
                window_hits[name].append(1 if hit else 0)
                
                # Calculate sliding window hit ratio
                if i >= window_size:
                    window = window_hits[name][-window_size:]
                    hit_ratio = sum(window) / window_size
                    results[name].append(hit_ratio)
        
        return results


def calculate_delta(hit_ratio_proposed: float, hit_ratio_baseline: float) -> float:
    """
    Calculate Delta metric: relative improvement of baseline over proposed.
    
    Delta(j, c, Proposed) = (HR_proposed - HR_j) / HR_proposed * 100
    
    Positive values mean Proposed is better.
    """
    if hit_ratio_proposed == 0:
        return 0.0
    return ((hit_ratio_proposed - hit_ratio_baseline) / hit_ratio_proposed) * 100
