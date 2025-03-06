from typing import Callable
from .predictor import Predictor
from .query_planner import QueryPlanner

def do_blind_attack(oracle: Callable, predictor: Predictor, queryplanner: QueryPlanner, known_prefix: str, known_suffix: str, logging=True):
  prefix = known_prefix
  if logging:
    print(f"Initial prefix: {prefix}")
  while not prefix.endswith(known_suffix):
    probs = predictor.predict(prefix)
    if logging:
      print({k: round(v, 4) for k, v in sorted(probs.items(), key=lambda item: item[1], reverse=True)})
    while 1 < len(probs):
      query, post_probs = queryplanner.plan(probs)
      if logging:
        print(f"{query=}, {len(probs)=}")
      result = oracle(prefix, query)
      # print(f"Oracle result: {result}")
      probs = post_probs[result]
    next_char = list(probs.keys())[0]
    if logging:
     print(f"Next character to add: {next_char}")
    prefix += next_char
    if logging:
      print(f"Updated prefix: {prefix}")
  return prefix
