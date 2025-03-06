from typing import Dict, List, Tuple, TypeVar
from heapq import heappush, heappop, heapify
from collections import defaultdict

QueryT = str
ResT = bool

class QueryPlanner:
    def plan(self, probs: Dict[str, float]) -> Tuple[QueryT, Dict[ResT, Dict[str, float]]]:
        raise NotImplementedError

class PrefixQueryPlanner(QueryPlanner):
    def plan(self, probs: Dict[str, float]) -> Tuple[QueryT, Dict[ResT, Dict[str, float]]]:
        query = max(probs, key=lambda k: probs[k])
        post_probs = {True: {query: probs[query]}, False: {k: v for k, v in probs.items() if k != query}}
        return query, post_probs

class ContainsQueryPlanner(QueryPlanner):
    def plan(self, probs: Dict[str, float]) -> Tuple[QueryT, Dict[ResT, Dict[str, float]]]:
        heap: List[Tuple[float, str]] = [(weight, char) for char, weight in probs.items()]
        heapify(heap)

        while len(heap) > 2:
            w1, c1 = heappop(heap)
            w2, c2 = heappop(heap)
            heappush(heap, (w1 + w2, c1 + c2))
        
        c1, c2 = heap[0][1], heap[1][1]
        if len(c1) > len(c2): c1, c2 = c2, c1
        return c1, { True: { c: probs[c] for c in c1 }, False: { c: probs[c] for c in c2 } }
