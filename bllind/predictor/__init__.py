import string
from typing import Dict


class Predictor:
  def predict(self, prefix: str) -> Dict[str, float]:
    raise NotImplementedError

  def filter(self, charset: str):
    from .filteredpredictor import FilteredPredictor

    return FilteredPredictor(self, charset)

  def as_ascii_printable(self, whitespaces=" "):
    return self.filter(
      string.digits + string.ascii_letters + string.punctuation + whitespaces
    )

  def as_alnum(self, whitespaces=" "):
    return self.filter(string.digits + string.ascii_letters)
