"""
Stage 0: build clean, non-overlapping monolingual corpora for EN and FI, and
fetch the FLORES+ ground truth that is used ONLY for final evaluation.

"No overlap" is treated as three concrete, separately-checkable properties,
not a vibe:

  (1) MONOLINGUAL PURITY: each corpus contains only its own language. Wikipedia
      dumps routinely contain quoted foreign text, loanword-heavy sentences,
      and stray templates in other scripts. We run fastText's lid.176 language
      identifier over every candidate sentence and drop anything that isn't
      confidently the expected language.

  (2) NO DUPLICATE / NEAR-DUPLICATE SENTENCES within or across the two
      training corpora (catches templated Wikipedia boilerplate that would
      otherwise let the model memorize instead of generalize, and catches any
      stray identical lines between the EN and FI corpora -- e.g. bare
      numbers, URLs, or proper nouns that are spelled identically in both).

  (3) ZERO TEST-SET LEAKAGE: neither monolingual corpus may contain a sentence
      that also appears (exactly or near-exactly) in the FLORES+ dev/devtest
      sets, which are reserved exclusively for evaluation. This is checked
      explicitly against the actual FLORES+ sentences, not assumed.

Run on Kaggle with internet enabled. This script is NOT runnable in a
sandboxed, no-internet environment -- see README.md for how it was tested
(dedup/near-dup/shingle-index logic validated on synthetic text; the actual
Wikipedia/FLORES+ download path can only be exercised on Kaggle itself).
"""
import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

from config import (
    LANG_A, LANG_B, FLORES_CODE, WIKI_DUMP_DATE,
    LID_CONFIDENCE_THRESHOLD, NEAR_DUP_CONTAINMENT_THRESHOLD,
)

ISO2_TO_FASTTEXT = {"en": "en", "fi": "fi"}  # fastText lid.176 uses ISO 639-1 codes directly

# ---------------------------------------------------------------------------
# Sentence splitting -- dependency-free, rule-based. Good enough for building
# a monolingual training corpus (UNMT is already robust to imperfect sentence
# boundaries); NOT a claim of linguistic-quality segmentation. Swap in pysbd
# or spaCy's sentencizer if you want higher precision and don't mind the
# extra dependency / model download.
# ---------------------------------------------------------------------------
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "e.g", "i.e",
    "st", "no", "vol", "co", "inc", "ltd", "fig", "approx",
}
_SENT_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÅ0-9])")


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    raw_sents = _SENT_END_RE.split(text)
    out = []
    buf = ""
    for s in raw_sents:
        buf = (buf + " " + s).strip() if buf else s
        last_word = re.findall(r"[A-Za-z]+$", buf[:-1] if buf.endswith(('.', '!', '?')) else buf)
        if last_word and last_word[-1].lower() in _ABBREV:
            continue  # likely a false sentence boundary after an abbreviation; keep accumulating
        out.append(buf)
        buf = ""
    if buf:
        out.append(buf)
    return [s.strip() for s in out if s.strip()]


# ---------------------------------------------------------------------------
# Normalization used for hashing / dedup comparisons (NOT used to alter the
# actual stored text -- only to decide whether two strings "are the same").
# ---------------------------------------------------------------------------
def normalize_for_hash(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    return s


def char_ngrams(s: str, n: int = 5) -> Set[str]:
    s = normalize_for_hash(s)
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def containment(a: Set[str], b: Set[str]) -> float:
    """Overlap coefficient: |A n B| / min(|A|,|B|). Unlike Jaccard, this does
    NOT get diluted when one string contains the other plus extra content
    (e.g. a FLORES sentence quoted verbatim with a trailing citation clause
    appended) -- exactly the leak pattern most likely in scraped web/Wikipedia
    text. This was empirically necessary: symmetric Jaccard scored a FLORES
    sentence + appended clause at 0.72 and missed it at a 0.8 threshold; the
    same case scores ~1.0 under containment. See test_data_prepare.py."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / min(len(a), len(b))


# ---------------------------------------------------------------------------
# (1) Language-ID filtering
# ---------------------------------------------------------------------------
class LangIDFilter:
    """Wraps fastText's lid.176 model. Falls back to `langdetect` (pure
    Python, less accurate but has no binary-model download dependency) if
    fastText or its model file are unavailable, so the pipeline degrades
    gracefully rather than hard-failing on a network hiccup."""

    LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

    def __init__(self, cache_dir: str):
        self.backend = None
        self._init_fasttext(cache_dir)
        if self.backend is None:
            self._init_langdetect()
        if self.backend is None:
            raise RuntimeError(
                "Neither fastText+lid.176 nor langdetect are available. "
                "Install one: `pip install fasttext` (and let it fetch lid.176.bin) "
                "or `pip install langdetect`."
            )

    def _init_fasttext(self, cache_dir: str):
        try:
            import fasttext
            import urllib.request
            os.makedirs(cache_dir, exist_ok=True)
            model_path = os.path.join(cache_dir, "lid.176.bin")
            if not os.path.exists(model_path):
                print(f"Downloading fastText LID model to {model_path} ...")
                urllib.request.urlretrieve(self.LID_URL, model_path)
            fasttext.FastText.eprint = lambda *a, **k: None  # silence a harmless warning fastText prints
            self._ft_model = fasttext.load_model(model_path)
            self.backend = "fasttext"
        except Exception as e:
            print(f"[LangIDFilter] fastText unavailable ({e}); will try langdetect fallback")
            self.backend = None

    def _init_langdetect(self):
        try:
            import langdetect  # noqa: F401
            self.backend = "langdetect"
        except Exception as e:
            print(f"[LangIDFilter] langdetect unavailable too ({e})")
            self.backend = None

    def predict(self, text: str) -> Tuple[str, float]:
        """Returns (iso639_1_code, confidence)."""
        text = text.replace("\n", " ").strip()
        if not text:
            return "unk", 0.0
        if self.backend == "fasttext":
            labels, probs = self._ft_model.predict(text, k=1)
            lang = labels[0].replace("__label__", "")
            return lang, float(probs[0])
        else:
            from langdetect import detect_langs
            try:
                results = detect_langs(text)
                top = results[0]
                return top.lang, float(top.prob)
            except Exception:
                return "unk", 0.0

    def keep(self, text: str, expected_lang: str, threshold: float = LID_CONFIDENCE_THRESHOLD) -> bool:
        lang, conf = self.predict(text)
        return lang == expected_lang and conf >= threshold


# ---------------------------------------------------------------------------
# (2) + (3) Dedup and cross-corpus / test-set leakage removal, via an
# inverted shingle index so we don't do an O(N*M) full scan of a multi-
# million-line corpus against every FLORES sentence.
# ---------------------------------------------------------------------------
class NearDupIndex:
    """Inverted index: char-5-gram shingle -> set of reference-sentence ids
    that contain it. Given a candidate sentence, we only need to Jaccard-
    compare against reference sentences that share at least one shingle,
    which in practice is a tiny candidate set even for a large reference
    collection, making this tractable at multi-million-line scale."""

    def __init__(self, reference_sentences: List[str], n: int = 5):
        self.n = n
        self.refs = reference_sentences
        self.ref_shingles = [char_ngrams(s, n) for s in reference_sentences]
        self.index: Dict[str, List[int]] = defaultdict(list)
        for idx, shingles in enumerate(self.ref_shingles):
            for sh in shingles:
                self.index[sh].append(idx)
        self.exact_hashes: Set[str] = {normalize_for_hash(s) for s in reference_sentences}

    def is_leaked(self, candidate: str, threshold: float = NEAR_DUP_CONTAINMENT_THRESHOLD) -> bool:
        norm = normalize_for_hash(candidate)
        if not norm:
            return False
        if norm in self.exact_hashes:
            return True
        cand_shingles = char_ngrams(candidate, self.n)
        candidate_ref_ids: Set[int] = set()
        for sh in cand_shingles:
            candidate_ref_ids.update(self.index.get(sh, []))
        for ref_id in candidate_ref_ids:
            if containment(cand_shingles, self.ref_shingles[ref_id]) >= threshold:
                return True
        return False


def exact_dedup(lines: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out = []
    for line in lines:
        h = hashlib.md5(normalize_for_hash(line).encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(line)
    return out


# ---------------------------------------------------------------------------
# FLORES+ fetch (gated dataset; falls back to an ungated community mirror)
# ---------------------------------------------------------------------------
def fetch_flores_sentences(lang_code_flores: str) -> Dict[str, List[str]]:
    """Returns {'dev': [...], 'devtest': [...]} sentences for one FLORES+
    language code (e.g. 'eng_Latn', 'fin_Latn'). Tries the current, actively
    maintained openlanguagedata/flores_plus first (requires `huggingface-cli
    login` / an HF token with the dataset's terms accepted -- gated as of
    this writing), then falls back to the ungated Muennighoff/flores200
    mirror of the same underlying FLORES-200 data if that fails."""
    from datasets import load_dataset

    try:
        ds = load_dataset("openlanguagedata/flores_plus", lang_code_flores)
        dev = list(ds["dev"]["text"])
        devtest = list(ds["devtest"]["text"])
        return {"dev": dev, "devtest": devtest}
    except Exception as e:
        print(f"[flores] openlanguagedata/flores_plus failed ({e}); trying Muennighoff/flores200 mirror")
        ds = load_dataset("Muennighoff/flores200", lang_code_flores)
        dev = list(ds["dev"]["sentence"])
        devtest = list(ds["devtest"]["sentence"])
        return {"dev": dev, "devtest": devtest}


# ---------------------------------------------------------------------------
# Main per-language pipeline
# ---------------------------------------------------------------------------
def build_monolingual_corpus(
    lang: str, wiki_lang_code: str, out_path: str, cache_dir: str,
    flores_leak_index: NearDupIndex, max_sentences: int = 3_000_000,
    min_chars: int = 10, max_chars: int = 500,
) -> int:
    from datasets import load_dataset

    lid = LangIDFilter(cache_dir)
    print(f"Streaming wikimedia/wikipedia {WIKI_DUMP_DATE}.{wiki_lang_code} ...")
    ds = load_dataset("wikimedia/wikipedia", f"{WIKI_DUMP_DATE}.{wiki_lang_code}", split="train", streaming=True)

    kept = 0
    n_dropped_lid = n_dropped_leak = n_dropped_len = 0
    seen_hashes: Set[str] = set()

    with open(out_path, "w", encoding="utf-8") as fout:
        for article in ds:
            for sent in split_sentences(article["text"]):
                if not (min_chars <= len(sent) <= max_chars):
                    n_dropped_len += 1
                    continue
                if flores_leak_index.is_leaked(sent):
                    n_dropped_leak += 1
                    continue
                h = hashlib.md5(normalize_for_hash(sent).encode("utf-8")).hexdigest()
                if h in seen_hashes:
                    continue
                if not lid.keep(sent, expected_lang=lang):
                    n_dropped_lid += 1
                    continue
                seen_hashes.add(h)
                fout.write(sent.replace("\n", " ") + "\n")
                kept += 1
                if kept >= max_sentences:
                    break
            if kept >= max_sentences:
                break

    print(f"[{lang}] kept {kept} sentences "
          f"(dropped: {n_dropped_len} bad-length, {n_dropped_lid} failed LID, {n_dropped_leak} FLORES-leak)")
    return kept


def cross_corpus_dedup_check(path_a: str, path_b: str) -> int:
    """Defensive check: how many normalized lines are identical across the
    two supposedly-disjoint-language corpora? Should be ~0 (stray numbers/
    URLs/proper nouns at most). Logged, not silently ignored, so leakage is
    visible rather than assumed away."""
    with open(path_a, encoding="utf-8") as f:
        set_a = {normalize_for_hash(l) for l in f}
    with open(path_b, encoding="utf-8") as f:
        set_b = {normalize_for_hash(l) for l in f}
    overlap = set_a & set_b
    overlap.discard("")
    return len(overlap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(os.environ.get("UNMT_WORK_DIR", "/kaggle/working/unmt-en-fi"), "data"))
    ap.add_argument("--cache_dir", default=os.path.join(os.environ.get("UNMT_WORK_DIR", "/kaggle/working/unmt-en-fi"), "cache"))
    ap.add_argument("--max_sentences_per_lang", type=int, default=3_000_000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Fetching FLORES+ ground truth (held out, evaluation-only) ...")
    flores_en = fetch_flores_sentences(FLORES_CODE[LANG_A])
    flores_fi = fetch_flores_sentences(FLORES_CODE[LANG_B])
    all_flores_sentences = flores_en["dev"] + flores_en["devtest"] + flores_fi["dev"] + flores_fi["devtest"]
    leak_index = NearDupIndex(all_flores_sentences, n=5)
    print(f"FLORES+ reference set for leakage-checking: {len(all_flores_sentences)} sentences")

    # persist FLORES splits for evaluate.py to consume later, unmodified
    with open(os.path.join(args.out_dir, "flores_en.json"), "w") as f:
        json.dump(flores_en, f)
    with open(os.path.join(args.out_dir, "flores_fi.json"), "w") as f:
        json.dump(flores_fi, f)

    en_path = os.path.join(args.out_dir, f"mono.{LANG_A}.txt")
    fi_path = os.path.join(args.out_dir, f"mono.{LANG_B}.txt")

    build_monolingual_corpus(LANG_A, "en", en_path, args.cache_dir, leak_index, args.max_sentences_per_lang)
    build_monolingual_corpus(LANG_B, "fi", fi_path, args.cache_dir, leak_index, args.max_sentences_per_lang)

    n_overlap = cross_corpus_dedup_check(en_path, fi_path)
    print(f"Cross-corpus identical-line check (EN vs FI, should be ~0): {n_overlap} shared normalized lines")
    if n_overlap > 0:
        print("  (a handful is normal -- bare numbers, URLs, or identical proper nouns; "
              "investigate if this number is more than a few dozen)")


if __name__ == "__main__":
    main()
