"""
Synthetic trace generators for cache evaluation.
"""

import numpy as np
from typing import Optional


class TraceGenerator:
    """Generate synthetic access traces for cache evaluation."""
    
    @staticmethod
    def zipf_stationary(num_items: int, num_requests: int, alpha: float, 
                        seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a stationary trace following Zipf distribution.
        
        Args:
            num_items: Number of unique items
            num_requests: Total number of requests to generate
            alpha: Zipf distribution parameter (higher = more skewed)
            seed: Random seed for reproducibility
        
        Returns:
            Array of item IDs representing the access trace
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate Zipf probabilities
        ranks = np.arange(1, num_items + 1)
        probabilities = 1.0 / (ranks ** alpha)
        probabilities /= probabilities.sum()
        
        # Generate trace
        trace = np.random.choice(num_items, size=num_requests, p=probabilities)
        return trace
    
    @staticmethod
    def non_stationary_phases(num_items: int, num_phases: int, phase_length: int,
                              alpha: float, p_hot: float, 
                              seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a non-stationary trace with phase changes.
        
        In each phase, a subset of items (hot set) is popular.
        The hot set changes between phases.
        
        Args:
            num_items: Total number of unique items
            num_phases: Number of phases
            phase_length: Number of requests per phase
            alpha: Zipf parameter within the hot set
            p_hot: Probability of accessing the hot set (vs cold set)
            seed: Random seed for reproducibility
        
        Returns:
            Array of item IDs representing the access trace
        """
        if seed is not None:
            np.random.seed(seed)
        
        trace = []
        hot_set_size = num_items // 4  # 25% of items are hot in each phase
        
        for phase in range(num_phases):
            # Determine hot set for this phase (different for each phase)
            hot_start = (phase * hot_set_size) % (num_items - hot_set_size)
            hot_items = np.arange(hot_start, hot_start + hot_set_size)
            cold_items = np.array([i for i in range(num_items) if i not in hot_items])
            
            # Generate Zipf probabilities for hot items
            ranks = np.arange(1, len(hot_items) + 1)
            hot_probs = 1.0 / (ranks ** alpha)
            hot_probs /= hot_probs.sum()
            
            # Uniform distribution for cold items
            cold_probs = np.ones(len(cold_items)) / len(cold_items)
            
            # Generate requests for this phase
            for _ in range(phase_length):
                if np.random.random() < p_hot:
                    item = np.random.choice(hot_items, p=hot_probs)
                else:
                    item = np.random.choice(cold_items, p=cold_probs)
                trace.append(item)
        
        return np.array(trace)
