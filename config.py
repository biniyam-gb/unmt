"""
Central configuration for the English<->Finnish unsupervised NMT system.
"""

import os
from dataclasses import dataclass

LANG_A = "en"
LANG_B = "fi"
LANG_IDS = {LANG_A: 0, LANG_B: 1}
FLORES_CODE = {LANG_A: "eng_Latn", LANG_B: "fin_Latn"}
WIKI_DUMP_DATE = "20231101"

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]

VOCAB_SIZE = 8000
CHAR_COVERAGE = 0.9998


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 512
    n_heads: int = 8
    n_enc_layers: int = 5
    n_dec_layers: int = 5
    d_ff: int = 2048
    dropout: float = 0.1
    max_len: int = 128
    n_langs: int = 2


MODEL_CFG = ModelConfig()

MIN_TOKENS_PER_SENT = 3
MAX_TOKENS_PER_SENT = 100
LID_CONFIDENCE_THRESHOLD = 0.85
NEAR_DUP_CONTAINMENT_THRESHOLD = 0.85

EMB_ALIGN_DIM = 512
EMB_TOP_FREQ_FOR_ADVERSARIAL = 20000
PROCRUSTES_REFINE_ITERS = 5
CSLS_K = 10

MAX_TOKENS_PER_BATCH = 6000
LR_SCALE = 1.0
WARMUP_STEPS = 4000
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0

DAE_STEPS = 1000
BT_STEPS = 2000
CHECKPOINT_EVERY_STEPS = 1000
CHECKPOINT_EVERY_SECONDS = 20 * 60

WORD_DROP_PROB = 0.1
SHUFFLE_WINDOW_K = 3

BEAM_SIZE = 5
LENGTH_PENALTY_ALPHA = 0.6

WORK_DIR = os.environ.get("UNMT_WORK_DIR", "/kaggle/working/unmt-en-fi")
