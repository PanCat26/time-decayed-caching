"""
Least Recently Used (LRU) cache replacement policy.
"""

from collections import OrderedDict
from cache_base import CachePolicy


class LRUCache(CachePolicy):
    """Least Recently Used (LRU) cache replacement policy."""
    
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.cache = OrderedDict()
    
    def access(self, item: int) -> bool:
        self.time += 1
        if item in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(item)
            self.hits += 1
            return True
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity:
                # Remove least recently used (first item)
                self.cache.popitem(last=False)
            self.cache[item] = True
            return False
    
    def reset(self):
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.time = 0
