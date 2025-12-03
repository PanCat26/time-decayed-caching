"""
Base class for cache replacement policies.
"""

from abc import ABC, abstractmethod
from typing import Dict


class CachePolicy(ABC):
    """Abstract base class for cache replacement policies."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
        self.time = 0
    
    @abstractmethod
    def access(self, item: int) -> bool:
        """
        Access an item in the cache.
        Returns True if hit, False if miss.
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the cache to initial state."""
        pass
    
    def get_hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_ratio': self.get_hit_ratio()
        }
