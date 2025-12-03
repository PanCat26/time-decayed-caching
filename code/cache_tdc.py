"""
Time-Decayed Caching (TDC) - Proposed Policy

Combines recency and frequency using exponential time decay.
"""

from cache_base import CachePolicy


class TimeDecayedCache(CachePolicy):
    """
    Time-Decayed Caching (TDC) - Proposed Policy
    
    Combines recency and frequency using exponential time decay.
    Score(item) = sum over accesses of: decay_rate^(current_time - access_time)
    
    The item with the lowest score is evicted when cache is full.
    """
    
    def __init__(self, capacity: int, decay_rate: float = 0.99):
        """
        Initialize TDC cache.
        
        Args:
            capacity: Maximum number of items in cache
            decay_rate: Decay factor (0 < decay_rate < 1), higher = slower decay
                       - Close to 1: More weight on frequency (like LFU)
                       - Close to 0: More weight on recency (like LRU)
        """
        super().__init__(capacity)
        self.decay_rate = decay_rate
        self.cache = {}  # item -> score
        self.last_update_time = {}  # item -> last update time
    
    def _update_score(self, item: int) -> float:
        """Update and return the current score for an item."""
        if item not in self.cache:
            return 0.0
        
        # Apply time decay to existing score
        time_diff = self.time - self.last_update_time[item]
        decayed_score = self.cache[item] * (self.decay_rate ** time_diff)
        return decayed_score
    
    def access(self, item: int) -> bool:
        self.time += 1
        
        if item in self.cache:
            # Hit: update score with decay and add 1 for this access
            current_score = self._update_score(item)
            self.cache[item] = current_score + 1.0
            self.last_update_time[item] = self.time
            self.hits += 1
            return True
        else:
            # Miss
            self.misses += 1
            
            if len(self.cache) >= self.capacity:
                # Evict item with minimum score (after applying decay)
                min_score = float('inf')
                min_item = None
                
                for cached_item in self.cache:
                    score = self._update_score(cached_item)
                    self.cache[cached_item] = score
                    self.last_update_time[cached_item] = self.time
                    
                    if score < min_score:
                        min_score = score
                        min_item = cached_item
                
                if min_item is not None:
                    del self.cache[min_item]
                    del self.last_update_time[min_item]
            
            # Add new item with initial score of 1
            self.cache[item] = 1.0
            self.last_update_time[item] = self.time
            return False
    
    def reset(self):
        self.cache = {}
        self.last_update_time = {}
        self.hits = 0
        self.misses = 0
        self.time = 0
