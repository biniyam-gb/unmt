"""
Central configuration for the English<->Finnish unsupervised NMT system.
Every other script imports from here so hyperparameters live in exactly one place.
"""
from dataclasses import dataclass, field

# ---- Language pair --------------------------------------------------------
LANG_A = "en"                 # English
LANG_B = "fi"                 # Finnish
LANG_IDS = {LANG_A: 0, LANG_B: 1}
FLORES_CODE = {LANG_A: "eng_Latn", LANG_B: "fin_Latn"}
# HF Wikipedia dump config names (wikimedia/wikipedia), confirmed available as of the
# 20231101 snapshot at dataset-creation time. If this exact dump date is ever removed,
# swap to the latest date listed at https://huggingface.co/datasets/wikimedia/wikipedia
WIKI_DUMP_DATE = "20231101"

# ---- Special tokens (SentencePiece ids) -----------------------------------
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]

# ---- Tokenizer -------------------------------------------------------------
VOCAB_SIZE = 32000
CHAR_COVERAGE = 0.9998        # high coverage needed for Finnish diacritics (ä, ö, š, ž)

# ---- Model architecture -----------------------------------------------------
@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 512
    n_heads: int = 8
    n_enc_layers: int = 5
    n_dec_layers: int = 5
    d_ff: int = 2048
    dropout: float = 0.1
    max_len: int = 128          # max source/target sentence length in subword tokens
    n_langs: int = 2

MODEL_CFG = ModelConfig()

# ---- Data filtering ---------------------------------------------------------
MIN_TOKENS_PER_SENT = 3
MAX_TOKENS_PER_SENT = 100
LID_CONFIDENCE_THRESHOLD = 0.85   # fastText lid.176 prob threshold to keep a line
NEAR_DUP_CONTAINMENT_THRESHOLD = 0.85  # char-5-gram overlap-coefficient above this = treat candidate as containing/being a leaked FLORES sentence

# ---- Stage A: unsupervised embedding alignment ------------------------------
EMB_ALIGN_DIM = 512                # must equal d_model (we init the model's embedding table directly)
EMB_TOP_FREQ_FOR_ADVERSARIAL = 20000   # also used as the top_k for the self-learning init (see align_embeddings.py)
PROCRUSTES_REFINE_ITERS = 5
CSLS_K = 10

# ---- Stage B/C: training -----------------------------------------------------
MAX_TOKENS_PER_BATCH = 6000        # dynamic batching budget PER GPU
# LR_SCALE: noam_lr_lambda() below ALREADY computes the complete target learning
# rate from Vaswani et al.'s formula (peak ~6.99e-4 for d_model=512, warmup=4000
# -- a standard, sensible peak LR on its own). LR_SCALE is an OPTIONAL further
# multiplier on top of that complete schedule, for if you deliberately want to
# scale the whole curve up or down; 1.0 means "use the paper's schedule exactly,"
# which is the correct default. An earlier version of this file had a separate
# LR=3e-4 constant that also got multiplied in via LambdaLR's own base_lr
# mechanism, silently composing two full learning-rate values together and
# shrinking the real, effective LR by ~3300x -- discovered from a real training
# log where loss stayed flat for 900+ steps. If you're tuning this, multiply
# LR_SCALE, don't reintroduce a second absolute rate.
LR_SCALE = 1.0
WARMUP_STEPS = 4000
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0
# DAE_STEPS / BT_STEPS: I have NOT benchmarked this model on a T4, so these are
# NOT calibrated to any real throughput measurement -- treat them as "large
# enough to run for a while," not as a considered target. Run
# profile_throughput.py on Kaggle first; it measures actual sec/step for DAE
# and BT on your hardware and prints a realistic step budget for a 12h session
# and for the 30h/week quota. Override these via --max_steps on the training
# scripts once you have that number, rather than trusting the defaults below.
DAE_STEPS = 60000
BT_STEPS = 150000
CHECKPOINT_EVERY_STEPS = 1000
CHECKPOINT_EVERY_SECONDS = 20 * 60  # also checkpoint on a wall-clock timer (Kaggle 12h hard cap)

# ---- Noise model (for denoising autoencoding) --------------------------------
WORD_DROP_PROB = 0.1
SHUFFLE_WINDOW_K = 3

# ---- Decoding ------------------------------------------------------------------
BEAM_SIZE = 5
LENGTH_PENALTY_ALPHA = 0.6

# ---- Paths (override via environment/CLI on Kaggle: everything persists under
# /kaggle/working so it survives until the notebook session ends; push it to a
# Kaggle Dataset if you want it to survive across sessions/quota resets) --------
import os
WORK_DIR = os.environ.get("UNMT_WORK_DIR", "/kaggle/working/unmt-en-fi")
