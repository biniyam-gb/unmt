# Unsupervised English <-> Finnish Neural Machine Translation

A complete, from-scratch unsupervised NMT (UNMT) system: two disjoint monolingual
corpora in, a trained translation model out, **zero parallel data, zero seed
dictionary, zero weak supervision of any kind** anywhere in training. Ground
truth (FLORES+) is used **exclusively** for final evaluation and is never seen
during training -- this is checked explicitly by code, not assumed.

## Why English <-> Finnish specifically

Finnish is Uralic; English is Indo-European/Germanic -- genuinely unrelated
families, not a "cheat" pair like English-French or English-German where
shared vocabulary and word order do a lot of unacknowledged work. Finnish is
agglutinative with 15 grammatical cases and vowel harmony, has no grammatical
gender, and marks negation with an inflecting verb rather than a particle --
structurally about as different from English as two Latin-script languages
get. It also has enough Wikipedia volume to be tractable on a 2xT4 budget, and
a clean, professionally-translated FLORES+ devtest set, which is what actually
made rigorous, leakage-free evaluation possible without inventing test data.

## The one honest calibration point, stated once

Guzman et al. (2019, "Two New Evaluation Datasets for Low-Resource Machine
Translation: Nepali-English and Sinhala-English") ran exactly this class of
method (Lample/Artetxe-style UNMT) on genuinely low-resource, typologically
distant pairs and got near-zero BLEU -- the "embedding spaces are
approximately isomorphic across languages" assumption this whole pipeline
rests on can simply fail to hold when the data is too thin. Finnish has vastly
more monolingual data than Nepali or Sinhala, so I'd expect this to produce a
genuinely working system rather than collapse outright -- but "working" here
means "meaningfully better than nothing, well below supervised NMT quality,"
not "competitive with Google Translate." Confidence: moderate, based on
published trends for this method family, not a claim about this specific run
(I haven't trained it to completion -- see "What was actually tested" below).

## Pipeline

| Stage | Script | What it does |
|---|---|---|
| 0 | `data_prepare.py` | Download EN+FI Wikipedia, sentence-split, LID-filter for monolingual purity, dedup, remove any overlap with FLORES+ |
| 0 | `train_tokenizer.py` | Train ONE joint SentencePiece model on both filtered corpora |
| 0 | `binarize.py` | Tokenize once, pack into memmap-able shards + per-token frequency counts |
| A | `run_stage_a.py` | Monolingual skip-gram embeddings -> self-learning seed dictionary -> iterative Procrustes/CSLS alignment -> combined init matrix for the shared embedding table |
| B | `train_dae.py` | Denoising-autoencoder pretraining (per-language, shared weights) |
| C | `train_bt.py` | Online back-translation -- the actual unsupervised MT training |
| - | `evaluate.py` | Beam-search decode + chrF2/spBLEU on FLORES+ devtest (held out, evaluation-only) |

Supporting modules: `config.py` (all hyperparameters, one place), `model.py`
(shared Transformer encoder-decoder + language embeddings + greedy/cached/beam
decoding), `align_embeddings.py` (CSLS, Procrustes, self-learning init),
`noise.py` (DAE noise model), `batching.py` (token-budget dynamic batching),
`utils_dist.py` (DDP + checkpointing), `profile_throughput.py` (measures your
actual step time before you commit to a long run).

## Why self-learning init, not adversarial (Conneau et al. 2018's MUSE method)

The obvious choice for Stage A is the adversarial (GAN-style) aligner from
Conneau, Lample et al. 2018. **I built it, tested it, and it doesn't reliably
converge to the correct alignment.** `test_alignment_synthetic.py` constructs
a synthetic embedding space related to another by a KNOWN planted rotation
and checks whether each candidate method recovers it. Adversarial training
converged to a confidently wrong rotation: discriminator accuracy climbed
above 90% while the recovered mapping moved *away* from the true rotation.

To isolate *why* -- bad math, or just a bad starting point? -- I ran a
separate, throwaway diagnostic (not saved in the test file, since it only
makes sense as a debugging step, not a real test): I handed the Procrustes
solver 200 word pairs read directly from the synthetic test's own answer key
-- something only possible because I constructed that synthetic data myself
and know the true correspondence, and something that has **no equivalent
anywhere in the real system**, which runs on Wikipedia text with no answer
key at all. Given that cheating seed, Procrustes + CSLS recovered the true
rotation to under 1% relative error. That confirmed the Procrustes/CSLS math
itself was correct, and the bug was specifically in the adversarial step's
ability to *find* a good starting point on its own -- not in what happens
once it has one.

So the adversarial step is replaced with something that finds a seed with
**zero supervision of any kind**: Artetxe, Labaka & Agirre's (ACL 2018)
self-learning initialization, which matches EN and FI subwords by comparing
the *shape* of each word's similarity profile to its own language's
vocabulary -- a descriptor computable from purely monolingual statistics,
with no cross-lingual information whatsoever. `test_alignment_synthetic.py`
validates exactly this version (function `similarity_profile_seed_dictionary`
feeding into `align_embedding_spaces`) and confirms it recovers the planted
rotation with no seed, no oracle, and no adversarial training. This is the
only version that actually runs: `align_embedding_spaces()` in
`align_embeddings.py` is what `run_stage_a.py` calls, and its function
signature takes only the two embedding matrices -- no seed dictionary
parameter exists for it to take even if you wanted to supply one. The broken
adversarial code (`adversarial_align()`) is kept in the file for reference
and comparison, clearly marked as not the default path, and is never called
by anything in the training pipeline.

## "No overlap," made concrete

Three separately-checkable properties, all implemented as code in
`data_prepare.py`, not asserted:

1. **Monolingual purity**: every line in each corpus is checked with
   fastText's `lid.176` language identifier; anything not confidently the
   expected language is dropped (Wikipedia dumps routinely contain quoted
   foreign text, loanword-heavy sentences, stray templates).
2. **No duplicate/near-duplicate sentences** within or across the two
   training corpora, via exact-hash dedup plus a shingle-indexed containment
   check (see next point for why containment, not Jaccard).
3. **Zero FLORES+ leakage**: both training corpora are checked against every
   FLORES+ dev/devtest sentence, using an inverted character-5-gram shingle
   index so this is tractable at multi-million-line scale (not an O(N x M)
   scan). Near-duplicate detection uses the **overlap coefficient**
   (containment: `|A n B| / min(|A|,|B|)`), not Jaccard -- this was an actual
   bug caught during testing: Jaccard under-detects a leaked sentence that has
   *extra* text appended (e.g. a FLORES sentence quoted with a trailing
   citation clause), because the union grows while the leaked content's share
   of it shrinks. A FLORES sentence + appended clause scored 0.72 Jaccard
   (missed at the 0.8 threshold I'd originally set) but ~1.0 containment.
   See `test_alignment_synthetic.py`'s sibling tests in the "what was tested"
   section below for the concrete before/after numbers.

## Setup on Kaggle

1. New Notebook -> Settings -> Accelerator: **GPU T4 x2**. Settings ->
   Internet: **on** (needed for `pip install`, Wikipedia, FLORES+).
2. Upload this whole directory, or `git clone`/copy it into
   `/kaggle/working/unmt-en-fi`.
3. `pip install -r requirements.txt --break-system-packages` (or without the
   flag if your image doesn't need it).
4. Run stages **in order**:

```bash
export UNMT_WORK_DIR=/kaggle/working/unmt-en-fi

# Stage 0: data (takes a while -- Wikipedia streaming + LID filtering)
python data_prepare.py --max_sentences_per_lang 3000000
python train_tokenizer.py
python binarize.py

# Measure YOUR actual throughput before committing to a step count
python profile_throughput.py

# Stage A: embedding alignment (CPU-bound, no GPU needed, a few hours for
# the skip-gram training depending on corpus size)
python run_stage_a.py

# Stage B: DAE pretraining (use the step count profile_throughput.py suggested
# for your session/quota budget, not the config.py default -- see below)
torchrun --nproc_per_node=2 train_dae.py --max_steps <YOUR_NUMBER>

# Stage C: online back-translation (bootstraps from the DAE checkpoint
# automatically; also re-run with a higher --max_steps to resume)
torchrun --nproc_per_node=2 train_bt.py --max_steps <YOUR_NUMBER>

# Evaluation (only place FLORES+ ground truth is used)
python evaluate.py --split devtest
```

5. **Kaggle will hard-kill your GPU session at 12 hours**, and your weekly GPU
   quota is ~30 hours (both current as of this writing -- verify against
   Kaggle's current docs since these limits are Kaggle's to change). Every
   training script checkpoints every 1000 steps AND every 20 minutes of wall
   clock, and **auto-resumes** from `checkpoints/{dae,bt}_latest.pt` if it
   exists -- just re-run the same command in a new session. Plan on spreading
   Stage B and Stage C across multiple sessions/weeks; that's expected, not a
   failure mode.

## On the step-count defaults in `config.py`

I have **not** benchmarked this exact model on a real T4. `DAE_STEPS=60000`
and `BT_STEPS=150000` in `config.py` are placeholders, not a considered
recommendation -- don't trust them. Run `profile_throughput.py` on your actual
Kaggle session first; it runs ~20 real DAE steps and ~20 real BT steps on your
hardware, measures actual sec/step, and prints how many steps fit in a 12-hour
session and in the full 30-hour weekly quota. Pass that number via
`--max_steps`. This is a deliberate choice: I'd rather hand you a tool that
measures the true number on your hardware than assert a specific figure I
can't verify from here.

One thing I *can* tell you with more confidence: BT steps are substantially
more expensive than DAE steps, because each BT step generates synthetic
translations (autoregressive, no free parallelism across the sequence
dimension) in *both* directions before training on them. I implemented
KV-caching for this generation step specifically because of that cost (see
next section) -- without it, back-translation would very likely be the
dominant cost of the entire pipeline.

## An efficiency fix that mattered enough to implement properly

Greedy decoding without KV-caching recomputes the *entire* growing sequence
from scratch at every generation step. For a length-T generated sequence,
that's O(T^3) total attention compute and O(T^2) total feedforward compute,
against O(T^2) and O(T) respectively with caching -- a roughly T-fold waste on
both components (T is ~20-40 for typical sentences here). This isn't a
rounding error: generation runs on **every single Stage C training step**, not
just at evaluation time, so this inefficiency would very plausibly have
dominated total BT training cost.

I implemented cached incremental decoding (`generate_greedy_cached` in
`model.py`) by hand-unrolling `nn.TransformerDecoderLayer`'s pre-LN forward
pass (verified against its actual source in this environment, not from
memory) and caching each layer's raw hidden state per position. Because a
caching change should be a pure speed optimization that changes nothing about
the output, I validated it the only way that actually proves that:
`test_kv_cache_equivalence.py` checks bit-for-bit identical output against the
uncached path across 3 model configs x 5 seeds x 3 batch sizes x variable
padding patterns (45 cases, all pass). Measured speedup on this sandbox's CPU
at a moderate model size was 3.2x; the asymptotic argument above says the gap
should be larger for longer sequences and I'd expect at least as large a
relative win on a T4, though I haven't measured that directly.

## What was actually tested (and what wasn't)

This sandbox has no GPU and no access to huggingface.co, Wikipedia dumps, or
the FLORES+ hosts (network egress is restricted to a small allowlist), so I
could not run the real download -> train -> evaluate pipeline end to end on
real data. What I *did* do, rather than just write code and assert it works:

- **Every module was tested against synthetic or toy data in this sandbox**,
  including full smoke tests of `train_dae.py` and `train_bt.py` (fresh run
  and checkpoint-resume, both bootstrapping-from-DAE and resuming-BT-itself)
  and `evaluate.py`, all via a tiny end-to-end toy EN/FI corpus and a tiny
  model config, run completely from scratch as a final regression check
  after all fixes.
- **Five real bugs were found and fixed this way**, not zero:
  1. `model.py`: mixing a bool padding mask with a float causal mask triggered
     a PyTorch deprecation warning that could silently break in a future
     torch version -- switched to an all-bool masking convention.
  2. `align_embeddings.py`: the adversarial aligner does not reliably
     converge to the correct alignment (see the dedicated section above) --
     replaced as the default path with a self-learning initializer, with the
     failure mode kept reproducible in `test_alignment_synthetic.py` rather
     than quietly deleted.
  3. `data_prepare.py`: Jaccard similarity under-detects near-duplicate leaks
     when the leaked sentence has extra content appended -- switched to the
     overlap/containment coefficient.
  4. `batching.py`: an editing mistake deleted the plain (non-noised) batch
     iterator that `train_bt.py` depends on -- caught immediately by
     re-running the batching test suite, not left for later.
  5. `utils_dist.py` / `train_dae.py`: resuming the LR scheduler by manually
     calling `.step()` in a loop (rather than saving/restoring its own
     `state_dict()`) triggered a PyTorch order-of-operations warning and would
     have been needlessly slow at real step counts (e.g. looping 140,000
     times just to restore state) -- fixed to persist scheduler state
     directly.
- **What remains genuinely unverified**: actual training convergence and
  final BLEU/chrF2 numbers on the real EN/FI Wikipedia data, actual T4
  throughput and DDP behavior across two real GPUs, and whether the FLORES+
  gated-dataset access path (`openlanguagedata/flores_plus`, needs an
  accepted-terms HF token) works smoothly in a Kaggle notebook context versus
  needing the ungated `Muennighoff/flores200` fallback that's already wired
  in. These can only be checked by actually running it on Kaggle -- which is
  exactly what this system is built for you to do next.

## Expected output quality (honest framing)

Do not expect this to be usable for anything you'd actually rely on. Realistic
outcomes for a compute-constrained, from-scratch UNMT system on a genuinely
distant language pair: substantially better than random/copy-the-input
baselines, capable of getting the gist of simple, common-vocabulary sentences
right some of the time, and unreliable on anything syntactically complex,
rare-vocabulary, or ambiguous. If `evaluate.py`'s chrF2 comes back in, very
roughly, the 25-40 range, treat that as "the unsupervised bootstrap is
working, in the range the literature would predict for a compute-limited run
on a distant, non-tiny-resource pair" rather than "translation quality
comparable to a production system." If it comes back near 0, check the
qualitative examples `evaluate.py` prints -- degenerate repeated-token output
usually means Stage B/C needs more steps, not that the method has failed
outright (compare against the loss curves logged during training).
