from functools import cache
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from numpy import ndarray
from platformdirs import user_cache_dir

from bllind.predictor import Predictor
from bllind.logger import logger

import torch
from tokenizers import Tokenizer
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  PreTrainedModel,
  PreTrainedTokenizer,
  PreTrainedTokenizerFast,
)
import re
import pickle
import time

CTF_FLAG_PROMPT = """
A CTF Flag usually forms a sentence encoded in a Leetspeak form, and words are joined with `_` or `-`, or sometimes ` `. In some cases charactors may be UPPER, but usually most of charactors are lowercases.

### Leetspeak (1337) Examples:
- `hello world` -> `h3l10_w0r1d`
- `flag found` -> `f14g_f0und`
- `secure code` -> `s3cur3_c0d3`
- `leet hacker` -> `1337_h4ck3r`

Here is an example of CTF flags:
- Flag{{h3ll0_w0rld}}
- {prefix}
""".strip()
DEFAULT_MODEL = "openai-community/gpt2"

global_cache_dir = Path(user_cache_dir("bllind"))


def canonicalize_model_name(model: str):
  return re.sub(r"[^\w-]", "-", model.lower())


def get_cache_path(
  cache_name: str,
  prompt: str | Callable[[str], str],
  model: str | Any,
  cache_slug: Optional[str],
):
  if cache_slug is None:
    if cache_name is not None and model is not None and prompt is not None:
      digest = hashlib.md5(f"{cache_name}##{model}##{prompt}".encode()).hexdigest()
    else:
      digest = None
  else:
    if "/" in cache_slug or "\\" in cache_slug:
      raise ValueError("cache_slug should not contain '/' or '\\'")
    digest = cache_slug

  if digest is None:
    raise ValueError("cache slug requires when the prompt or model is not a string")
  else:
    return (
      global_cache_dir / f"{cache_name}_{canonicalize_model_name(model)}_{cache_slug}"
    )


cached_model_retriever = cache(AutoModelForCausalLM.from_pretrained)
cached_tokenizer_retriever = cache(AutoTokenizer.from_pretrained)


class LLMPredictor(Predictor):
  def __init__(
    self,
    prompt: str | Callable[[str], str],
    model: str | PreTrainedModel = DEFAULT_MODEL,
    tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast] = None,
    use_attention_cache: bool = True,
    permanent_next_token_cache: bool = False,
    cache_slug: Optional[str] = None,
  ):
    self.promptGetter = (
      prompt if callable(prompt) else lambda prefix: prompt.format(prefix=prefix)
    )
    if isinstance(model, str):
      logger.info("loading the model and the tokenizer...")
      self.model: PreTrainedModel = cached_model_retriever(model)
      self.tokenizer = cached_tokenizer_retriever(model)
    else:
      if tokenizer is None:
        raise ValueError("Tokenizer must be provided when model object is provided")
      self.model = model
      self.tokenizer = tokenizer
    self.model.eval()

    self.token_count = len(self.tokenizer)
    self.tokens = self.tokenizer.convert_ids_to_tokens(list(range(self.token_count)))
    assert isinstance(self.tokens, list)

    self.next_token_cache_path = (
      get_cache_path("next_token", prompt, model, cache_slug)
      if permanent_next_token_cache
      else None
    )

    self.use_attention_cache = use_attention_cache
    self.past_key_values: Dict[str, Dict[str, float]] = {}

    self.next_tokens: Dict[str, Dict[str, float]] = {}
    self.next_char_probs_dict: Dict[str, Dict[str, float]] = {}

    self.token_decoder: Callable[[str], str] = lambda s: s.replace("Ġ", " ").replace(
      "▁", " "
    )

  def _get_permanent_next_token_cache(self, prefix: str):
    if self.next_token_cache_path is None:
      return None
    path = self.next_token_cache_path / hashlib.md5(prefix.encode()).hexdigest()
    if not path.exists():
      return None
    logger.debug("permanent next_token cache hit!")
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
            self.token_decoder,
            self.tokenizer.convert_ids_to_tokens(input_ids[0][:-1]),  # type: ignore
          )
        )
        past_kv = self.past_key_values.get(prev_input)
        logger.debug(f"has attention cache: {past_kv is not None}")
      else:
        past_kv = None
      with torch.no_grad():
        start_time = time.time()
        if past_kv is None:
          outputs = self.model(**inputs)
        else:
          outputs = self.model(
            input_ids[:, -1:],  # type: ignore
            past_key_values=past_kv,
          )
          # assert self.model(**inputs).logits == outputs.logits
        end_time = time.time()
        logger.debug(f"Time taken for prediction: {end_time - start_time:.2f} secs")
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
      token_value = 1
      cur_prefix = prediction_prefix
      for c in grown_prefix:
        token_value *= self.next_char_probs_dict[cur_prefix][c]
        cur_prefix += c

      for token, prob in predicted_result.items():
        if len(token) <= len(grown_prefix) or not token.startswith(grown_prefix):
          continue
        first_char = token[len(grown_prefix)]
        if first_char not in next_char_probs:
          next_char_probs[first_char] = 0
        next_char_probs[first_char] += prob / token_value

    total_prob = sum(next_char_probs.values())
    if total_prob > 0:
      next_char_probs = {
        char: prob / total_prob for char, prob in next_char_probs.items()
      }

    self.next_char_probs_dict[prefix] = next_char_probs

    return next_char_probs
