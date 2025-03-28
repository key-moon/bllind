from bllind.benchmark import benchmark
from bllind.predictor.llmpredictor import LLMPredictor

# from bllind.logger import logger
# from logging import WARN
# logger.setLevel(WARN)

MISTRAL = "mistralai/Mistral-7B-v0.1"
GPT2 = "openai-community/gpt2"
PHI_2 = "microsoft/phi-2"
GEMMA3 = "google/gemma-3-1b-it"

prompt = """
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


# # gemma3 requires a login
# from huggingface_hub import login
# login()

benchmark(
  lambda: LLMPredictor(
    prompt=prompt,
    model=GEMMA3,
    permanent_next_token_cache=True,
    use_attention_cache=True,
  ).as_ascii_printable(),
  [
    """Flag{https://github.com/golang/go/issues/59968}""",
    """Flag{1_r3ally_d0nt_w4nt_t0_us3_g3t5_197b38d9}""",
  ],
  known_prefix="Flag{",
  known_suffix="}",
)
