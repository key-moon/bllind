
from typing import Callable, Dict

from bllind.predictor import Predictor

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CTF_FLAG_PROMPT = "A CTF Flag usually forms a sentence encoded in a 1337 form, and words are joined with `-` or `_`. Following is a example: {prefix}"
DEFAULT_MODEL = "openai-community/gpt2"
# DEFAULT_MODEL = "TinyLlama/TinyLlama_v1.1"
# DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

class LLMPredictor(Predictor):
    def __init__(self, prompt: str | Callable[[str], str], model: str=DEFAULT_MODEL):
        self.promptGetter = prompt if callable(prompt) else lambda prefix: prompt.format(prefix=prefix)
        self.model = AutoModelForCausalLM.from_pretrained(model); self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        
        self.token_count = len(self.tokenizer)
        self.tokens = self.tokenizer.convert_ids_to_tokens(list(range(self.token_count)))

        self.succeded_tokens = {}

    def predict(self, prefix: str) -> Dict[str, float]:
        prompt = self.promptGetter(prefix)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits[:, -1, :], dim=-1)
            token_probabilities = probabilities[0].cpu().numpy()
        
        first_char_probs = {}
        for token, prob in zip(self.tokens, token_probabilities):
            if not token:
                continue
            first_char = token[0]
            if first_char not in first_char_probs:
                first_char_probs[first_char] = 0
            first_char_probs[first_char] += float(prob)

        return first_char_probs
