from pathlib import Path
from transformers import AutoTokenizer

from interpretable_nlp.models import ModelTypes

SENTIMENT_MODEL = ModelTypes.SBERBANK_RU_ROBERTA_LARGE

TOKENIZER_PATH = SENTIMENT_MODEL.value
TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
PAD_IDX = TOKENIZER.pad_token_id
CLS_IDX = TOKENIZER.cls_token_id
SEP_IDX = TOKENIZER.sep_token_id

VOCAB_SIZE = TOKENIZER.vocab_size
MAX_LEN = 512

