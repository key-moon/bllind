from functools import cache
import hashlib
from pathlib import Path
from typing import Callable, Dict, Union

from numpy import ndarray
from platformdirs import user_cache_dir

from bllind.predictor import Predictor

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import pickle
import time

CTF_FLAG_PROMPT = "A CTF Flag usually forms a sentence encoded in a 1337 form, and words are joined with `-` or `_`. Following is a example: {prefix}"
DEFAULT_MODEL = "openai-community/gpt2"
# DEFAULT_MODEL = "TinyLlama/TinyLlama_v1.1"
# DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

global_cache_dir = Path(user_cache_dir("bllind"))


def canonicalize_model_name(model: str):
  return re.sub(r"[^\w-]", "-", model.lower())


def get_cache_path(
  cache: Union[bool, str],
  cache_name: str,
  prompt: str | Callable[[str], str],
  model: str,
):
  if isinstance(cache, bool):
    if cache:
      if not isinstance(prompt, str):
        raise ValueError("cache name requires when the prompt is not a string")
      digest = hashlib.md5(f"{cache_name}##{model}##{prompt}".encode()).hexdigest()
      return (
        global_cache_dir / f"{cache_name}_{canonicalize_model_name(model)}_{digest}"
      )
    else:
      return None

  if "/" not in cache:
    digest = hashlib.md5(
      f"{cache_name}##{model}##{cache}##{prompt}".encode()
    ).hexdigest()
    return (
      global_cache_dir
      / f"{cache_name}_{canonicalize_model_name(model)}_{cache}_{digest}"
    )

  user_cache_dir = Path(cache)
  return user_cache_dir


cached_model_retriever = cache(AutoModelForCausalLM.from_pretrained)
cached_tokenizer_retriever = cache(AutoTokenizer.from_pretrained)

DEBUG_CACHE = True


class LLMPredictor(Predictor):
  def __init__(
    self,
    prompt: str | Callable[[str], str],
    model: str = DEFAULT_MODEL,
    use_attention_cache: bool = True,
    permanent_next_token_cache: Union[bool, str] = False,
  ):
    self.promptGetter = (
      prompt if callable(prompt) else lambda prefix: prompt.format(prefix=prefix)
    )
    self.model = cached_model_retriever(model)
    self.model.eval()
    self.tokenizer = cached_tokenizer_retriever(model)

    self.token_count = len(self.tokenizer)
    self.tokens = self.tokenizer.convert_ids_to_tokens(list(range(self.token_count)))

    self.next_token_cache_path = get_cache_path(
      permanent_next_token_cache, "next_token", prompt, model
    )

    self.use_attention_cache = use_attention_cache
    self.past_key_values: Dict[str, Dict[str, float]] = {}

    self.next_tokens: Dict[str, Dict[str, float]] = {}
    self.next_char_probs_dict: Dict[str, Dict[str, float]] = {}

    self.token_decoder: Callable[[str], str] = lambda s: s.replace("Ġ", " ")

  def _get_permanent_next_token_cache(self, prefix: str):
    if self.next_token_cache_path is None:
      return None
    path = self.next_token_cache_path / hashlib.md5(prefix.encode()).hexdigest()
    if not path.exists():
      return None
    print("[*] permanent next_token cache hit!")
    with path.open("rb") as f:
      return pickle.load(f)

  def _set_permanent_next_token_cache(self, prefix: str, content):
    if self.next_token_cache_path is None:
      return
    self.next_token_cache_path.mkdir(parents=True, exist_ok=True)
    path = self.next_token_cache_path / hashlib.md5(prefix.encode()).hexdigest()
    with path.open("wb") as f:
      pickle.dump(content, f)

  def predict(self, prefix: str) -> Dict[str, float]:
    if prefix in self.next_char_probs_dict:
      return self.next_char_probs_dict[prefix]

    prompt = self.promptGetter(prefix)
    inputs = self.tokenizer(prompt, return_tensors="pt")

    next_token_probs = self._get_permanent_next_token_cache(prefix)
    if next_token_probs is None:
      input_ids = inputs["input_ids"]
      if self.use_attention_cache:
        prev_input = "".join(
          map(
            self.token_decoder, self.tokenizer.convert_ids_to_tokens(input_ids[0][:-1])
          )
        )
        past_kv = self.past_key_values.get(prev_input)
        print(f"[*] has attention cache: {past_kv is not None}")
      else:
        past_kv = None
      with torch.no_grad():
        start_time = time.time()
        if past_kv is None:
          outputs = self.model(**inputs)
        else:
          outputs = self.model(
            input_ids[:, -1:],
            past_key_values=past_kv,
          )
          # if DEBUG_CACHE:
          # assert self.model(**inputs).logits == outputs.logits
        end_time = time.time()
        print(f"[*] Time taken for prediction: {end_time - start_time:.2f} secs")
        if self.use_attention_cache:
          self.past_key_values[prompt] = outputs.past_key_values
        probabilities = torch.softmax(outputs.logits[:, -1, :], dim=-1)
        next_token_probs = probabilities[0].cpu().numpy()
      self._set_permanent_next_token_cache(prefix, next_token_probs)

    cur_next_token = {
      self.token_decoder(token): float(prob)
      for token, prob in zip(self.tokens, next_token_probs)
    }
    self.next_tokens[prefix] = cur_next_token

    next_char_probs = {}
    for prediction_prefix, predicted_result in self.next_tokens.items():
      if not prefix.startswith(prediction_prefix):
        continue

      grown_prefix = prefix[len(prediction_prefix) :]
      # FIXME: 価値は grown prefix の構築を達成できる確率である必要がある
      #        そういうケースはまれなので一旦無視
      if grown_prefix == "":
        token_value = 1
      else:
        if grown_prefix not in predicted_result:
          continue
        token_value = predicted_result[grown_prefix]

      for token, prob in predicted_result.items():
        if len(token) <= len(grown_prefix) or not token.startswith(grown_prefix):
          continue
        first_char = token[len(grown_prefix)]
        if first_char not in next_char_probs:
          next_char_probs[first_char] = 0
        next_char_probs[first_char] += prob / token_value

    self.next_char_probs_dict[prefix] = next_char_probs

    return next_char_probs
