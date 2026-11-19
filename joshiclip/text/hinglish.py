"""Devanagari -> readable Hinglish (Roman) for captions. Deterministic, no models.

1. Lexicon lookup (common Hinglish spellings, English loanwords, names) - exact match.
2. Otherwise rule-based transliteration with schwa deletion:
   - word-final inherent 'a' is dropped (कर -> kar), except single-consonant words (न -> na)
   - medial inherent 'a' is dropped in a V C [a] C V context (समझा -> samjha, लड़का -> ladka)
   - long aa (ा) is written 'aa' only in single-syllable words (यार -> yaar), else 'a'.
The lexicon is a plain dict so it can be extended from config without code changes.
"""
from __future__ import annotations

import re
import unicodedata

LEXICON: dict[str, str] = {
    # function words / very common
    "है": "hai", "हैं": "hain", "था": "tha", "थी": "thi", "थे": "the", "नहीं": "nahi", "नही": "nahi",
    "हां": "haan", "हाँ": "haan", "हा": "haan", "ना": "na", "तो": "to", "भी": "bhi", "ही": "hi", "को": "ko",
    "से": "se", "का": "ka", "की": "ki", "के": "ke", "में": "mein", "मैं": "main", "पे": "pe", "पर": "par",
    "ये": "ye", "यह": "ye", "वो": "vo", "वह": "vo", "क्या": "kya", "क्यों": "kyun", "कौन": "kaun", "कैसे": "kaise",
    "कहां": "kahan", "कहाँ": "kahan", "जब": "jab", "अब": "ab", "अभी": "abhi", "फिर": "phir", "और": "aur",
    "लेकिन": "lekin", "यार": "yaar", "भाई": "bhai", "अरे": "arey", "अबे": "abe", "ओ": "o", "हे": "hey",
    "इसको": "isko", "उसको": "usko", "इसका": "iska", "उसका": "uska", "इसने": "isne", "उसने": "usne",
    "मेरे": "mere", "मेरा": "mera", "मेरी": "meri", "तेरे": "tere", "तेरा": "tera", "तेरी": "teri",
    "हम": "hum", "तुम": "tum", "आप": "aap", "आपको": "aapko", "हमारा": "hamara", "हमारे": "hamare",
    "कुछ": "kuch", "बहुत": "bahut", "सब": "sab", "आज": "aaj", "कल": "kal", "बस": "bas", "सच": "sach",
    "अच्छा": "accha", "बढ़िया": "badhiya", "मस्त": "mast", "जैसे": "jaise", "जैसा": "jaisa", "शायद": "shayad",
    "रहा": "raha", "रही": "rahi", "रहे": "rahe", "गया": "gaya", "गई": "gayi", "गए": "gaye", "दिया": "diya",
    "करेगा": "karega", "करेगी": "karegi", "करता": "karta", "करती": "karti", "करो": "karo", "कर": "kar",
    "दो": "do", "दे": "de", "लो": "lo", "ले": "le", "हो": "ho", "हुआ": "hua", "हुई": "hui", "होगा": "hoga",
    "जाएंगे": "jaayenge", "जाएगा": "jaayega", "जाओ": "jao", "आओ": "aao", "आ": "aa", "जा": "ja",
    "देखो": "dekho", "देख": "dekh", "देखना": "dekhna", "रुको": "ruko", "चलो": "chalo", "बोल": "bol",
    "मिला": "mila", "मिलाए": "milaye", "मिलाना": "milana", "लड़ेगा": "ladega", "मजे": "maze", "मज़े": "maze",
    "इज्जत": "izzat", "इज़्ज़त": "izzat", "वाला": "wala", "वाली": "wali", "वाले": "wale", "कह": "keh",
    "गुस्सा": "gussa", "पापा": "Papa", "मम्मी": "Mummy", "जी": "ji", "दादा": "Dada", "दादी": "Dadi",
    "नाना": "Nana", "नानी": "Nani", "बेच": "bech", "बेचोगे": "bechoge", "पता": "pata", "लगता": "lagta",
    "पहचान": "pehchan", "पहचानते": "pehchante", "खाना": "khana", "कान": "kaan", "कुत्ता": "kutta",
    # names on the channel
    "ओरियो": "Oreo", "औरियो": "Oreo", "सौरव": "Sourav", "पीयूष": "Piyush", "पियूष": "Piyush",
    "साहिल": "Sahil", "कुणाली": "Kunali", "कुनाली": "Kunali", "टफी": "Toffee", "टॉफी": "Toffee",
    "अवंतिका": "Avantika", "ओलू": "Olu",
    # English loanwords as Whisper/YouTube spell them in Devanagari
    "गाइस": "guys", "गाइज़": "guys", "गाइज": "guys", "गुड": "good", "मॉर्निंग": "morning", "वेलकम": "welcome",
    "टू": "to", "माय": "my", "न्यू": "new", "वॉग": "vlog", "व्लॉग": "vlog", "वीडियो": "video",
    "फैमिली": "family", "रिएक्शन": "reaction", "मेल": "male", "फीमेल": "female", "प्रैंक": "prank",
    "रिप्लेस": "replace", "सरप्राइज": "surprise", "सरप्राइज़": "surprise", "डॉग": "dog", "कैमरा": "camera",
    "फोन": "phone", "बिकॉज़": "because", "बिकॉज": "because", "ओके": "okay", "सॉरी": "sorry", "थैंक": "thank",
    "यू": "you", "लाइक": "like", "सब्सक्राइब": "subscribe", "चैनल": "channel", "एडिटिंग": "editing",
    "लिफ्ट": "lift", "कार": "car", "गाड़ी": "gaadi", "फार्मर्स": "farmers", "मशीन": "machine",
}

V_IND = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ए": "e", "ऐ": "ai",
         "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e"}
MATRA = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "oo", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o",
         "ौ": "au", "ॉ": "o", "ॅ": "e"}
CONS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh",
        "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d",
        "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r",
        "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h",
        "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "फ़": "f", "ड़": "d", "ढ़": "dh", "य़": "y"}
NUKTA = "़"
VIRAMA = "्"
ANUSVARA, CHANDRA, VISARGA = "ं", "ँ", "ः"
LABIAL = set("पफबभम")
DEVA = re.compile(r"[ऀ-ॿ]")


def _units(word: str) -> list[dict]:
    """Split into units: {'c': consonant or None, 'v': vowel string or None (None = inherent schwa), 'nas': bool}."""
    word = unicodedata.normalize("NFC", word)
    out, i = [], 0
    while i < len(word):
        ch = word[i]
        if i + 1 < len(word) and word[i + 1] == NUKTA:
            ch = unicodedata.normalize("NFC", ch + NUKTA); i += 1
        if ch in CONS:
            u = {"c": CONS[ch], "v": None, "nas": False, "raw": ch, "half": False}
            nxt = word[i + 1] if i + 1 < len(word) else ""
            if nxt == VIRAMA:
                u["half"] = True; i += 1
            elif nxt in MATRA:
                u["v"] = MATRA[nxt]; u["matra"] = nxt; i += 1
            out.append(u)
        elif ch in V_IND:
            out.append({"c": None, "v": V_IND[ch], "nas": False, "raw": ch, "half": False})
        elif ch in (ANUSVARA, CHANDRA) and out:
            out[-1]["nas"] = True
        elif ch == VISARGA and out:
            out[-1]["vis"] = True
        i += 1
    return out


def _translit(word: str) -> str:
    u = _units(word)
    if not u:
        return word
    n = len(u)
    vowelled = [x["v"] is not None or (x["c"] is not None and not x["half"]) for x in u]
    syl = sum(1 for x in u if not x["half"])
    # schwa deletion
    keep = [True] * n
    for k in range(n - 1, -1, -1):
        x = u[k]
        if x["c"] is None or x["half"] or x["v"] is not None or x["nas"]:
            continue
        if k == n - 1:
            keep[k] = n == 1                            # final schwa dropped unless 1-consonant word
        elif 0 < k < n - 1 and vowelled[k - 1] and u[k + 1]["c"] is not None and keep[k + 1] is not False:
            nxt_has_vowel = u[k + 1]["v"] is not None or (k + 1 == n - 1 and False)
            if nxt_has_vowel:
                keep[k] = False
    s = []
    for k, x in enumerate(u):
        if x["c"]:
            s.append(x["c"])
        if x["half"]:
            continue
        v = x["v"]
        if v is None and x["c"]:
            v = "a" if keep[k] else ""
        if v == "a" and x.get("matra") == "ा" and syl == 1:
            v = "aa"
        if v == "i" and x.get("matra") == "ी" and syl == 1 and k < n - 1:
            v = "ee"
        if v == "oo" and k == n - 1 and syl > 1:
            v = "u"
        s.append(v or "")
        if x["nas"]:
            nxt = u[k + 1]["raw"] if k + 1 < n else ""
            s.append("m" if nxt in LABIAL else "n")
        if x.get("vis"):
            s.append("h")
    out = "".join(s)
    out = re.sub(r"aa(?=a)", "a", out)
    return out


def romanize_word(word: str) -> str:
    core = re.sub(r"[^ऀ-ॿ]", "", word)
    punct_tail = re.sub(r"[ऀ-ॿ]", "", word)
    if not core:
        return word
    core = core.rstrip("।")
    base = LEXICON.get(unicodedata.normalize("NFC", core)) or _translit(core)
    tail = "".join(ch for ch in punct_tail if ch in "?!,")
    return base + tail


def romanize(text: str) -> str:
    return " ".join(romanize_word(w) if DEVA.search(w) else w for w in text.split())
