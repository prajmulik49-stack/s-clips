"""Lexicon matching on romanized text via phonetic keys.

Whisper (language=hi) writes Devanagari, including English loanwords ("वेलकम टू माय व्लॉग"). We romanize
(hinglish.py) and compare *phonetic keys* (consonant skeletons) so "welcome"=="velkam", "guys"=="gaaiz".
"""
from __future__ import annotations

import functools
import re

from rapidfuzz import fuzz

from . import hinglish

_DIGRAPHS = [("chh", "C"), ("ch", "C"), ("sh", "S"), ("kh", "K"), ("gh", "G"), ("th", "t"), ("dh", "d"),
             ("bh", "b"), ("ph", "f"), ("jh", "J"), ("ck", "k")]
_SINGLE = str.maketrans({"c": "k", "w": "v", "z": "s", "q": "k", "x": "s"})
_LEX: dict = {}


def set_lexicon(lx: dict) -> None:
    _LEX.clear(); _LEX.update(lx or {})
    extra = (lx or {}).get("romanize") or {}
    hinglish.LEXICON.update(extra)
    key_word.cache_clear()


def lexicon() -> dict:
    return _LEX


@functools.lru_cache(maxsize=200_000)
def key_word(w: str) -> str:
    w = hinglish.romanize(w) if hinglish.DEVA.search(w) else w
    w = re.sub(r"[^a-z0-9]", "", w.lower())
    if not w:
        return ""
    w = re.sub(r"(?<=[aeiou])h$", "", w)
    for a, b in _DIGRAPHS:
        w = w.replace(a, b)
    head, rest = w[0], w[1:].translate(_SINGLE)
    head = "V" if head in "aeiou" else head.translate(_SINGLE)
    tail = re.sub(r"[aeiou]", "", rest)
    k = re.sub(r"(.)\1+", r"\1", head + tail)
    if len(k) <= 2:   # short words keep their first vowel so kya != ke != ki
        v = re.search(r"[aeiou]", rest)
        k += v.group(0) if v else ""
    return k


def keys(text: str) -> list[str]:
    return [k for k in (key_word(t) for t in text.split()) if k]


def phrase_in(phrase: str, text: str, ratio: int = 85) -> bool:
    pk = keys(phrase)
    tk = keys(text)
    if not pk or not tk:
        return False
    n = len(pk)
    for i in range(len(tk) - n + 1):
        if tk[i:i + n] == pk:
            return True
    if n >= 3:   # tolerate one inserted word ("welcome to my NEW vlog")
        for i in range(len(tk) - n):
            win = tk[i:i + n + 1]
            if win[0] == pk[0] and win[-1] == pk[-1] and any(win[:j] + win[j + 1:] == pk for j in range(1, n)):
                return True
    p, t = " ".join(pk), " ".join(tk)
    # the text must be at least as long as the phrase: partial_ratio otherwise finds a one-word sentence ("haan")
    # inside the phrase ("milte hain kal") and calls it a match
    if len(p) >= 6 and n >= 2 and len(t) >= len(p):
        return fuzz.partial_ratio(p, t) >= ratio
    return False


def hits(category: str, text: str, ratio: int = 85) -> list[str]:
    return [p for p in _flat(_LEX.get(category, [])) if phrase_in(p, text, ratio)]


def _flat(v):
    if isinstance(v, dict):
        return [x for vv in v.values() for x in _flat(vv)]
    return list(v)


def first_word_dependent(text: str) -> str | None:
    toks = text.split()
    if not toks:
        return None
    k = key_word(toks[0])
    for w in _LEX.get("dependent_first", []):
        if key_word(w) == k:
            return w
    return None


def sentence_complete(s: dict) -> bool:
    """Ends with terminal punctuation, or ends at a clear pause and not on a connective."""
    toks = s["text"].split()
    if not toks:
        return False
    last = key_word(toks[-1])
    dangling = {key_word(w) for w in _LEX.get("incomplete_end", [])}
    if last in dangling:
        return False
    return bool(s.get("punct_end")) or s.get("pause_after", 0) >= 0.5


def content_keys(text: str) -> list[str]:
    stop = {key_word(w) for w in _LEX.get("stopwords", [])}
    return [k for k in keys(text) if k not in stop and len(k) >= 2]
