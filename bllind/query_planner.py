from typing import Dict, List, Tuple, TypeVar
from heapq import heappush, heappop, heapify
from collections import defaultdict

QueryT = str
ResT = bool


class QueryPlanner:
  def plan(
    self, probs: Dict[str, float]
  ) -> Tuple[QueryT, Dict[ResT, Dict[str, float]]]:
    raise NotImplementedError


class PrefixQueryPlanner(QueryPlanner):
  def plan(
    self, probs: Dict[str, float]
  ) -> Tuple[QueryT, Dict[ResT, Dict[str, float]]]:
    query = max(probs, key=lambda k: probs[k])
    post_probs = {
      True: {query: probs[query]},
      False: {k: v for k, v in probs.items() if k != query},
    }
    return query, post_probs


class ContainsQueryPlanner(QueryPlanner):
  def plan(
    self, probs: Dict[str, float]
  ) -> Tuple[QueryT, Dict[ResT, Dict[str, float]]]:
    heap: List[Tuple[float, str]] = [(weight, char) for char, weight in probs.items()]
    heapify(heap)

    while len(heap) > 2:
      w1, c1 = heappop(heap)
      w2, c2 = heappop(heap)
      heappush(heap, (w1 + w2, c1 + c2))

    c1, c2 = heap[0][1], heap[1][1]
    if len(c1) > len(c2):
      c1, c2 = c2, c1
    return c1, {True: {c: probs[c] for c in c1}, False: {c: probs[c] for c in c2}}


class LessThanQueryPlanner(QueryPlanner):
  def plan(
    self, probs: Dict[str, float]
  ) -> Tuple[QueryT, Dict[ResT, Dict[str, float]]]:
    # TODO: add dp result caching
    items = sorted(probs.items())

    # p: prefix sums of probabilities
    p = [0.0]
    for _, w in items:
      p.append(p[-1] + w)

    n = len(items)

    # dp[i][j] := minimum cost to query the range items[i:j)
    dp = [[0.0] * (n + 1) for _ in range(n + 1)]
    # choose[i][j] := index of the optimal query point in the range items[i:j)
    choose = [[0] * (n + 1) for _ in range(n + 1)]

    # Fill dp and choose tables
    for length in range(2, n + 1):
      for i in range(n - length + 1):
        j = i + length
        dp[i][j] = float("inf")
        for k in range(i + 1, j):
          cost = dp[i][k] + dp[k][j] + (p[j] - p[i])
          if cost < dp[i][j]:
            dp[i][j] = cost
            choose[i][j] = k

    # The optimal query point for the entire range is stored in choose[0][n]
    idx = choose[0][n]

    return items[idx][0], {True: dict(items[:idx]), False: dict(items[idx:])}
