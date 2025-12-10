"""
Least Frequently Used (LFU) cache replacement policy.
"""

from collections import OrderedDict, defaultdict
from cache_base import CachePolicy


class LFUCache(CachePolicy):
    """Least Frequently Used (LFU) cache replacement policy."""
    
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.cache = {}  # item -> frequency
        self.freq_to_items = defaultdict(OrderedDict)  # frequency -> OrderedDict of items
        self.min_freq = 0
    
    def access(self, item: int) -> bool:
        self.time += 1
        if item in self.cache:
            # Update frequency
            freq = self.cache[item]
            del self.freq_to_items[freq][item]
            if not self.freq_to_items[freq]:
                del self.freq_to_items[freq]
                if self.min_freq == freq:
                    self.min_freq += 1
            
            self.cache[item] = freq + 1
            self.freq_to_items[freq + 1][item] = True
            self.hits += 1
            return True
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity:
                # Remove least frequently used
                lfu_items = self.freq_to_items[self.min_freq]
                evict_item = next(iter(lfu_items))
                del lfu_items[evict_item]
                if not lfu_items:
                    del self.freq_to_items[self.min_freq]
                del self.cache[evict_item]
            
            self.cache[item] = 1
            self.freq_to_items[1][item] = True
            self.min_freq = 1
            return False
    
    def reset(self):
        self.cache = {}
        self.freq_to_items = defaultdict(OrderedDict)
        self.min_freq = 0
        self.hits = 0
        self.misses = 0
        self.time = 0
