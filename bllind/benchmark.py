from typing import Callable, Dict, List, Tuple

from regex import R
from bllind.predictor import Predictor
from bllind.query_planner import (
  ContainsQueryPlanner,
  LessThanQueryPlanner,
  PrefixQueryPlanner,
)
from bllind.runner import do_blind_attack
import time

ORACLE_TYPE = ""


class BenchmarkResult:
  def __init__(self, answer: str):
    self.answer = answer
    self.predict_times = [0.0 for _ in range(len(self.answer) + 1)]
    self.queries_dict = {}

  def hook(self, oracle_type):
    self.queries_dict[oracle_type] = [[] for _ in range(len(self.answer) + 1)]

    def _hook(func):
      def _oracle(prefix, query):
        self.queries_dict[oracle_type][len(prefix)].append(query)
        return func(prefix, query)

      return _oracle

    return _hook

  def query_per_char(self, oracle_type):
    qs = [q for q in self.queries_dict[oracle_type] if q]
    if not qs:
      return -1
    return sum([len(q) for q in qs]) / len(qs)

  def elapsed_per_char(self):
    ts = [t for t in self.predict_times if t != 0]
    if not ts:
      return -1
    return sum([queries for queries in ts]) / len(ts)

  def dump(self):
    for oracle_type, queries in self.queries_dict.items():
      max_queries = max(len(queries) for queries in queries)
      print(f"{oracle_type}: {self.query_per_char(oracle_type):.2f} query / char")
      for qs in queries:
        query_count = len(qs)
        intensity = int(255 * (query_count / max_queries)) if max_queries > 0 else 0
        red = intensity
        green = 128 - int(intensity / 2)
        color = (red, green, 0)
        print(
          f"\033[48;2;{color[0]};{color[1]};{color[2]}m {query_count:2} \033[0m", end=""
        )
      print()
    MAX_TIME, MAGNIFICATION = 1, 1000
    print(
      f"prediction time: {self.elapsed_per_char():.2f} sec / char(total: {sum(self.predict_times):.2f} sec)"
    )
    for predict_time in self.predict_times:
      rounded_time = int(round(predict_time * MAGNIFICATION))
      intensity = (
        int(255 * min(rounded_time / (MAX_TIME * MAGNIFICATION), 1))
        if max_queries > 0
        else 0
      )
      red = intensity
      green = 128 - int(intensity / 2)
      color = (red, green, 0)
      print(
        f"\033[48;2;{color[0]};{color[1]};{color[2]}m{rounded_time:4}\033[0m", end=""
      )
    print()

    for char in self.answer + " ":
      print(f"  {char} ", end="")
    print()


class BenchmarkingPredictor(Predictor):
  def __init__(self, predictor: Predictor, result: BenchmarkResult):
    self.predictor = predictor
    self.result = result
    self.cache: Dict[str, Dict[str, float]] = {}

  def predict(self, prefix: str) -> Dict[str, float]:
    if prefix in self.cache:
      return self.cache[prefix]

    start_time = time.time()
    prediction = self.predictor.predict(prefix)
    end_time = time.time()
    self.result.predict_times[len(prefix)] += end_time - start_time
    self.cache[prefix] = prediction
    return prediction


def benchmark(predictor_gen: Callable[[], Predictor], dataset: List[str], **args):
  results = []
  for data in dataset:
    result = single_benchmark(predictor_gen(), data, **args)
    result.dump()
    results.append(result)

  prefix_avg_qpc = sum(res.query_per_char("prefix") for res in results) / len(dataset)
  contains_avg_qpc = sum(res.query_per_char("contains") for res in results) / len(
    dataset
  )
  # less_than_avg_qpc = sum(res.query_per_char("less_than") for res in results) / len(
  #   dataset
  # )

  print(f"Prefix average QPC: {prefix_avg_qpc:.2f}")
  print(f"Contains average QPC: {contains_avg_qpc:.2f}")
  # print(f"Less than average QPC: {less_than_avg_qpc:.2f}")


def single_benchmark(predictor: Predictor, answer: str, **args) -> BenchmarkResult:
  result = BenchmarkResult(answer)

  @result.hook("prefix")
  def prefix_oracle(prefix: str, next_char: str):
    return answer.startswith(prefix + next_char)

  @result.hook("contains")
  def contains_oracle(prefix: str, charset: str):
    return answer[len(prefix)] in charset

  # @result.hook("less_than")
  def less_than_oracle(prefix: str, charset: str):
    return answer < prefix + charset

  predictor = BenchmarkingPredictor(predictor, result)
  do_blind_attack(prefix_oracle, predictor, PrefixQueryPlanner(), **args)
  do_blind_attack(contains_oracle, predictor, ContainsQueryPlanner(), **args)
  # do_blind_attack(less_than_oracle, predictor, LessThanQueryPlanner(), **args)

  return result
