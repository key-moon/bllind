from typing import Dict
from bllind.predictor import Predictor


class FilteredPredictor(Predictor):
  def __init__(self, predictor: Predictor, charset: str, eps: float = 1e-10):
    self.predictor = predictor
    self.charset = charset
    self.eps = eps

  def predict(self, prefix: str) -> Dict[str, float]:
    base_result = self.predictor.predict(prefix)
    return {c: base_result.get(c, self.eps) for c in self.charset}
