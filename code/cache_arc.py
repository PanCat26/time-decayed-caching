"""
Adaptive Replacement Cache (ARC) policy.
"""

from collections import OrderedDict
from cache_base import CachePolicy


class ARCCache(CachePolicy):
    """
    Adaptive Replacement Cache (ARC) policy.
    Maintains two LRU lists: T1 (recency) and T2 (frequency).
    Also maintains ghost lists B1 and B2 for adaptation.
    """
    
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.c = capacity  # Cache capacity
        self.p = 0  # Target size for T1
        
        # T1: Pages seen only once recently
        self.t1 = OrderedDict()
        # T2: Pages seen at least twice recently
        self.t2 = OrderedDict()
        # B1: Ghost entries for T1
        self.b1 = OrderedDict()
        # B2: Ghost entries for T2
        self.b2 = OrderedDict()
    
    def _replace(self, in_b2: bool):
        """Replace a page from cache."""
        if self.t1 and ((in_b2 and len(self.t1) == self.p) or len(self.t1) > self.p):
            # Remove from T1 and add to B1
            old, _ = self.t1.popitem(last=False)
            self.b1[old] = True
            # Limit B1 size
            if len(self.b1) > self.c:
                self.b1.popitem(last=False)
        else:
            if self.t2:
                # Remove from T2 and add to B2
                old, _ = self.t2.popitem(last=False)
                self.b2[old] = True
                # Limit B2 size
                if len(self.b2) > self.c:
                    self.b2.popitem(last=False)
    
    def access(self, item: int) -> bool:
        self.time += 1
        
        # Case 1: Hit in T1 or T2
        if item in self.t1:
            del self.t1[item]
            self.t2[item] = True
            self.hits += 1
            return True
        
        if item in self.t2:
            self.t2.move_to_end(item)
            self.hits += 1
            return True
        
        # Cache miss
        self.misses += 1
        
        # Case 2: Item in B1 (ghost hit)
        if item in self.b1:
            # Adapt: increase target size of T1
            delta = max(1, len(self.b2) // max(1, len(self.b1)))
            self.p = min(self.c, self.p + delta)
            
            self._replace(False)
            del self.b1[item]
            self.t2[item] = True
            return False
        
        # Case 3: Item in B2 (ghost hit)
        if item in self.b2:
            # Adapt: decrease target size of T1
            delta = max(1, len(self.b1) // max(1, len(self.b2)))
            self.p = max(0, self.p - delta)
            
            self._replace(True)
            del self.b2[item]
            self.t2[item] = True
            return False
        
        # Case 4: Complete miss
        l1 = len(self.t1) + len(self.b1)
        l2 = len(self.t2) + len(self.b2)
        
        if l1 == self.c:
            if len(self.t1) < self.c:
                self.b1.popitem(last=False)
                self._replace(False)
            else:
                self.t1.popitem(last=False)
        elif l1 < self.c and l1 + l2 >= self.c:
            if l1 + l2 >= 2 * self.c:
                if self.b2:
                    self.b2.popitem(last=False)
            self._replace(False)
        
        # Add to T1
        if len(self.t1) + len(self.t2) >= self.c:
            if self.t1:
                old, _ = self.t1.popitem(last=False)
                self.b1[old] = True
                if len(self.b1) > self.c:
                    self.b1.popitem(last=False)
            elif self.t2:
                old, _ = self.t2.popitem(last=False)
                self.b2[old] = True
                if len(self.b2) > self.c:
                    self.b2.popitem(last=False)
        
        self.t1[item] = True
        return False
    
    def reset(self):
        self.p = 0
        self.t1 = OrderedDict()
        self.t2 = OrderedDict()
        self.b1 = OrderedDict()
        self.b2 = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.time = 0
