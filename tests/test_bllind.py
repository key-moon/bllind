import unittest
from typing import Dict, Tuple
from bllind import do_blind_attack
from bllind.query_planner import QueryPlanner

from bllind.predictor.llmpredictor import Predictor

TEST_FLAG = "FLAG{test}"

class MockPredictor(Predictor):
    def predict(self, prefix: str) -> Dict[str, float]:
        return {char: 1.0 for char in TEST_FLAG}

class MockPlanner(QueryPlanner):
    def plan(self, probs: Dict[str, float]) -> Tuple[str, Dict[bool, Dict[str, float]]]:
        query = next(iter(probs))
        post_probs = {True: {query: probs[query]}, False: {k: v for k, v in probs.items() if k != query}}
        return query, post_probs

class TestBllind(unittest.TestCase):
    def test_do_blind_attack(self):
        def oracle(prefix: str, next_char: str):
            return TEST_FLAG.startswith(prefix + next_char)

        predictor = MockPredictor()
        queryplanner = MockPlanner()
        result = do_blind_attack(oracle, predictor, queryplanner, known_prefix="FLAG{", known_suffix="}")
        self.assertEqual(result, "FLAG{test}")

if __name__ == "__main__":
    unittest.main()