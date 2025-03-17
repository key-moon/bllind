import string
from logging import DEBUG
from bllind import do_blind_attack
from bllind.predictor.llmpredictor import LLMPredictor, CTF_FLAG_PROMPT
from bllind.query_planner import LessThanQueryPlanner

# comment out to see debug logs
# logger.logger.setLevel(DEBUG)

FLAG = "FLAG{blind_sql_injection}"


def oracle(prefix: str, next_char: str):
  return FLAG[len(prefix)] < next_char


GEMMA_3 = "google/gemma-3-1b-it"
PHI_2 = "microsoft/phi-2"

MODEL = GEMMA_3

print(
  do_blind_attack(
    oracle,
    LLMPredictor(prompt=CTF_FLAG_PROMPT, model=MODEL).filter(
      string.ascii_lowercase + "_{}"
    ),
    queryplanner=LessThanQueryPlanner(),
    known_prefix="FLAG{",
    known_suffix="}",
  )
)
