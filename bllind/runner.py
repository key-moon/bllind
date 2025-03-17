from typing import Callable
from .predictor import Predictor
from .query_planner import QueryPlanner
from .logger import logger


def do_blind_attack(
  oracle: Callable,
  predictor: Predictor,
  queryplanner: QueryPlanner,
  known_prefix: str,
  known_suffix: str,
):
  prefix = known_prefix
  logger.debug(f"Initial prefix: {prefix}")
  while not prefix.endswith(known_suffix):
    probs = predictor.predict(prefix)
    while 1 < len(probs):
      query, post_probs = queryplanner.plan(probs)
      logger.debug(f"Query: {query}, |candidates|: {len(probs)}")
      result = oracle(prefix, query)
      logger.debug(f"Oracle result: {result}")
      probs = post_probs[result]
    next_char = list(probs.keys())[0]
    logger.debug(f"Next character to add: {next_char}")
    prefix += next_char
    logger.debug(f"Updated prefix: {prefix}")
  return prefix
