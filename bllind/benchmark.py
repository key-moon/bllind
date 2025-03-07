from typing import Dict, List, Tuple
from bllind.predictor import Predictor
from bllind.query_planner import (
  ContainsQueryPlanner,
  LessThanQueryPlanner,
  PrefixQueryPlanner,
)
from bllind.runner import do_blind_attack

ORACLE_TYPE = ""


class CachedPredictor(Predictor):
  def __init__(self, predictor: Predictor):
    self.predictor = predictor
    self.cache: Dict[str, Dict[str, float]] = {}

  def predict(self, prefix: str) -> Dict[str, float]:
    if prefix in self.cache:
      return self.cache[prefix]

    prediction = self.predictor.predict(prefix)
    self.cache[prefix] = prediction
    return prediction


class BenchmarkResult:
  def __init__(self, answer: str):
    self.answer = answer
    self.queries = [[] for _ in range(len(answer) + 1)]

  def hook(self, func):
    def _oracle(prefix, query):
      self.queries[len(prefix)].append(query)
      return func(prefix, query)

    return _oracle

  def query_per_char(self):
    qs = [q for q in self.queries if q != 0]
    return sum([len(queries) for queries in qs]) / len(qs)

  def dump(self):
    max_queries = max(len(queries) for queries in self.queries)
    print(f"{self.query_per_char():.2f} query / char")
    for i, (queries, char) in enumerate(zip(self.queries, self.answer + " ")):
      query_count = len(queries)
      intensity = int(255 * (query_count / max_queries)) if max_queries > 0 else 0
      red = intensity
      green = 128 - int(intensity / 2)
      color = (red, green, 0)
      print(
        f"\033[48;2;{color[0]};{color[1]};{color[2]}m {query_count:2} \033[0m", end=""
      )
    print()
    for char in self.answer + " ":
      print(f"  {char} ", end="")
    print()


def benchmark(predictor: Predictor, dataset: List[str], **args):
  prefix_results, contains_results, less_than_results = [], [], []
  for data in dataset:
    prefix_result, contains_result, less_than_result = single_benchmark(
      predictor, data, **args
    )
    prefix_result.dump()
    contains_result.dump()
    less_than_result.dump()
    prefix_results.append(prefix_result)
    contains_results.append(contains_result)
    less_than_results.append(less_than_result)

  prefix_avg_qpc = sum(res.query_per_char() for res in prefix_results) / len(dataset)
  contains_avg_qpc = sum(res.query_per_char() for res in contains_results) / len(
    dataset
  )
  less_than_avg_qpc = sum(res.query_per_char() for res in less_than_results) / len(
    dataset
  )

  print(f"Prefix average QPC: {prefix_avg_qpc:.2f}")
  print(f"Contains average QPC: {contains_avg_qpc:.2f}")
  print(f"Less than average QPC: {less_than_avg_qpc:.2f}")


def single_benchmark(
  predictor: Predictor, answer: str, **args
) -> Tuple[BenchmarkResult, BenchmarkResult, BenchmarkResult]:
  prefix_result, contains_result, less_than_result = (
    BenchmarkResult(answer),
    BenchmarkResult(answer),
    BenchmarkResult(answer),
  )

  @prefix_result.hook
  def prefix_oracle(prefix: str, next_char: str):
    return answer.startswith(prefix + next_char)

  @contains_result.hook
  def contains_oracle(prefix: str, charset: str):
    return answer[len(prefix)] in charset

  @less_than_result.hook
  def less_than_oracle(prefix: str, charset: str):
    return answer < prefix + charset

  predictor = CachedPredictor(predictor)
  do_blind_attack(prefix_oracle, predictor, PrefixQueryPlanner(), **args)
  do_blind_attack(contains_oracle, predictor, ContainsQueryPlanner(), **args)
  do_blind_attack(less_than_oracle, predictor, LessThanQueryPlanner(), **args)

  return prefix_result, contains_result, less_than_result
