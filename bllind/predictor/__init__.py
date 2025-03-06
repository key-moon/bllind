import string
from typing import Dict

class Predictor:
    def predict(self, prefix: str) -> Dict[str, float]:
        raise NotImplementedError
    def as_ascii_printable(self, whitespaces=" "):
        from .filteredpredictor import FilteredPredictor
        charset = string.digits + string.ascii_letters + string.punctuation + whitespaces
        return FilteredPredictor(self, charset)
