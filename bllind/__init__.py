from typing import Callable
from .predictor import Predictor
from .query_planner import QueryPlanner

def do_blind_attack(oracle: Callable, predictor: Predictor, queryplanner: QueryPlanner, known_prefix: str, known_suffix: str):
  prefix = known_prefix
  print(f"Initial prefix: {prefix}")
  while not prefix.endswith(known_suffix):
    probs = predictor.predict(prefix)
    print({k: round(v, 4) for k, v in sorted(probs.items(), key=lambda item: item[1], reverse=True)})
    while 1 < len(probs):
      query, post_probs = queryplanner.plan(probs)
      print(f"{query=}, {len(probs)=}")
      result = oracle(prefix, query)
      # print(f"Oracle result: {result}")
      probs = post_probs[result]
    next_char = list(probs.keys())[0]
    print(f"Next character to add: {next_char}")
    prefix += next_char
    print(f"Updated prefix: {prefix}")
  return prefix
