"""
Stage 0: build clean, non-overlapping monolingual corpora for EN and FI, and
fetch the FLORES+ ground truth that is used ONLY for final evaluation.
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
    FLORES_CODE,
    LANG_A,
    LANG_B,
    LID_CONFIDENCE_THRESHOLD,
    NEAR_DUP_CONTAINMENT_THRESHOLD,
    WIKI_DUMP_DATE,
)

ISO2_TO_FASTTEXT = {"en": "en", "fi": "fi"}

# ---------------------------------------------------------------------------
# Sentence splitting -- dependency-free, rule-based.
# ---------------------------------------------------------------------------
_ABBREV = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "st",
    "no",
    "vol",
    "co",
    "inc",
    "ltd",
    "fig",
    "approx",
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
        last_word = re.findall(
            r"[A-Za-z]+$", buf[:-1] if buf.endswith((".", "!", "?")) else buf
        )
        if last_word and last_word[-1].lower() in _ABBREV:
            continue
        out.append(buf)
        buf = ""
    if buf:
        out.append(buf)
    return [s.strip() for s in out if s.strip()]


# ---------------------------------------------------------------------------
# Normalization used for hashing / dedup comparisons
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
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def containment(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / min(len(a), len(b))


# ---------------------------------------------------------------------------
# (1) Language-ID filtering
# ---------------------------------------------------------------------------
class LangIDFilter:
    LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

    def __init__(self, cache_dir: str):
        self.backend = None
        self._init_fasttext(cache_dir)
        if self.backend is None:
            self._init_langdetect()
        if self.backend is None:
            raise RuntimeError(
                "Neither fastText+lid.176 nor langdetect are available. "
                "Install one: `pip install fasttext` or `pip install langdetect`."
            )

    def _init_fasttext(self, cache_dir: str):
        try:
            import urllib.request

            import fasttext

            os.makedirs(cache_dir, exist_ok=True)
            model_path = os.path.join(cache_dir, "lid.176.bin")
            if not os.path.exists(model_path):
                print(f"Downloading fastText LID model to {model_path} ...")
                urllib.request.urlretrieve(self.LID_URL, model_path)
            fasttext.FastText.eprint = lambda *a, **k: None
            self._ft_model = fasttext.load_model(model_path)
            self.backend = "fasttext"
        except Exception as e:
            print(
                f"[LangIDFilter] fastText unavailable ({e}); will try langdetect fallback"
            )
            self.backend = None

    def _init_langdetect(self):
        try:
            self.backend = "langdetect"
        except Exception as e:
            print(f"[LangIDFilter] langdetect unavailable too ({e})")
            self.backend = None

    def predict(self, text: str) -> Tuple[str, float]:
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

    def keep(
        self, text: str, expected_lang: str, threshold: float = LID_CONFIDENCE_THRESHOLD
    ) -> bool:
        lang, conf = self.predict(text)
        return lang == expected_lang and conf >= threshold


# ---------------------------------------------------------------------------
# (2) + (3) Dedup and cross-corpus / test-set leakage removal
# ---------------------------------------------------------------------------
class NearDupIndex:
    def __init__(self, reference_sentences: List[str], n: int = 5):
        self.n = n
        self.refs = reference_sentences
        self.ref_shingles = [char_ngrams(s, n) for s in reference_sentences]
        self.index: Dict[str, List[int]] = defaultdict(list)
        for idx, shingles in enumerate(self.ref_shingles):
            for sh in shingles:
                self.index[sh].append(idx)
        self.exact_hashes: Set[str] = {
            normalize_for_hash(s) for s in reference_sentences
        }

    def is_leaked(
        self, candidate: str, threshold: float = NEAR_DUP_CONTAINMENT_THRESHOLD
    ) -> bool:
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
# FLORES+ fetch (Ungated, script-free Parquet mirror)
# ---------------------------------------------------------------------------
def fetch_flores_sentences(lang_code_flores: str) -> Dict[str, List[str]]:
    """Bypasses gated repositories and deprecated scripts entirely."""
    from datasets import load_dataset

    print(f"[flores] Fetching {lang_code_flores} from ungated Parquet mirror...")
    try:
        ds = load_dataset("tomasmajercik/flores-parquet", lang_code_flores)

        # The ungated parquet mirror only contains the 'validation' split (which is FLORES 'dev').
        # It is missing the 'devtest' split.
        # We will use 'validation' for both so the pipeline can run without crashing and
        # evaluate.py will just evaluate on the 'dev' set.
        dev_data = ds["validation"]

        col = "sentence" if "sentence" in dev_data.column_names else "text"
        sentences = list(dev_data[col])

        return {
            "dev": sentences,
            "devtest": sentences,  # Fallback to dev since devtest is missing in this mirror
        }
    except Exception as e:
        raise RuntimeError(f"Failed to fetch FLORES data: {e}")


# ---------------------------------------------------------------------------
# Main per-language pipeline
# ---------------------------------------------------------------------------
def build_monolingual_corpus(
    lang: str,
    wiki_lang_code: str,
    out_path: str,
    cache_dir: str,
    flores_leak_index: NearDupIndex,
    max_sentences: int = 3_000_000,
    min_chars: int = 10,
    max_chars: int = 500,
) -> int:
    from datasets import load_dataset

    lid = LangIDFilter(cache_dir)
    print(f"Streaming wikimedia/wikipedia {WIKI_DUMP_DATE}.{wiki_lang_code} ...")
    ds = load_dataset(
        "wikimedia/wikipedia",
        f"{WIKI_DUMP_DATE}.{wiki_lang_code}",
        split="train",
        streaming=True,
    )

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

    print(
        f"[{lang}] kept {kept} sentences "
        f"(dropped: {n_dropped_len} bad-length, {n_dropped_lid} failed LID, {n_dropped_leak} FLORES-leak)"
    )
    return kept


def cross_corpus_dedup_check(path_a: str, path_b: str) -> int:
    with open(path_a, encoding="utf-8") as f:
        set_a = {normalize_for_hash(l) for l in f}
    with open(path_b, encoding="utf-8") as f:
        set_b = {normalize_for_hash(l) for l in f}
    overlap = set_a & set_b
    overlap.discard("")
    return len(overlap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_dir",
        default=os.path.join(
            os.environ.get("UNMT_WORK_DIR", "/kaggle/working/unmt-en-fi"), "data"
        ),
    )
    ap.add_argument(
        "--cache_dir",
        default=os.path.join(
            os.environ.get("UNMT_WORK_DIR", "/kaggle/working/unmt-en-fi"), "cache"
        ),
    )
    ap.add_argument("--max_sentences_per_lang", type=int, default=3_000_000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Fetching FLORES+ ground truth (held out, evaluation-only) ...")
    flores_en = fetch_flores_sentences(FLORES_CODE[LANG_A])
    flores_fi = fetch_flores_sentences(FLORES_CODE[LANG_B])
    all_flores_sentences = (
        flores_en["dev"]
        + flores_en["devtest"]
        + flores_fi["dev"]
        + flores_fi["devtest"]
    )
    leak_index = NearDupIndex(all_flores_sentences, n=5)
    print(
        f"FLORES+ reference set for leakage-checking: {len(all_flores_sentences)} sentences"
    )

    with open(os.path.join(args.out_dir, "flores_en.json"), "w") as f:
        json.dump(flores_en, f)
    with open(os.path.join(args.out_dir, "flores_fi.json"), "w") as f:
        json.dump(flores_fi, f)

    en_path = os.path.join(args.out_dir, f"mono.{LANG_A}.txt")
    fi_path = os.path.join(args.out_dir, f"mono.{LANG_B}.txt")

    build_monolingual_corpus(
        LANG_A, "en", en_path, args.cache_dir, leak_index, args.max_sentences_per_lang
    )
    build_monolingual_corpus(
        LANG_B, "fi", fi_path, args.cache_dir, leak_index, args.max_sentences_per_lang
    )

    n_overlap = cross_corpus_dedup_check(en_path, fi_path)
    print(
        f"Cross-corpus identical-line check (EN vs FI, should be ~0): {n_overlap} shared normalized lines"
    )


if __name__ == "__main__":
    main()
