# readability_core.py — multilingual scoring engine for the SMOG-lights app.
# Kept free of Streamlit so it can be unit-tested and reused from scripts.
#
# textstat supplies the language-dependent building blocks (syllable counting via
# cmudict/pyphen, easy-word lists, Arabic counters). The formulas themselves are
# written out below from their published sources, all on one shared set of counts,
# so each can be checked line by line. See FORMULAS.md for the audit.
from __future__ import annotations

import math
import re
import statistics
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import pyphen
from textstat.textstat import textstatistics

# ----------------- Languages -----------------
@dataclass(frozen=True)
class Language:
    code: str
    name: str
    headline: Optional[str]       # default headline metric key
    word_based: bool = True       # False for scripts written without spaces (zh, ja)
    lingua: Optional[str] = None  # lingua.Language member name, for auto-detection

LANGUAGES: Dict[str, Language] = {l.code: l for l in [
    Language("en", "English", "smog_index", lingua="ENGLISH"),
    Language("es", "Spanish", "szigriszt_pazos", lingua="SPANISH"),
    Language("de", "German", "flesch_reading_ease", lingua="GERMAN"),
    Language("fr", "French", "flesch_reading_ease", lingua="FRENCH"),
    Language("it", "Italian", "gulpease_index", lingua="ITALIAN"),
    Language("nl", "Dutch", "flesch_reading_ease", lingua="DUTCH"),
    Language("pt", "Portuguese", "flesch_reading_ease", lingua="PORTUGUESE"),
    Language("ru", "Russian", "flesch_reading_ease", lingua="RUSSIAN"),
    Language("pl", "Polish", "pisarek", lingua="POLISH"),
    Language("hu", "Hungarian", "lix", lingua="HUNGARIAN"),
    Language("sv", "Swedish", "lix", lingua="SWEDISH"),
    Language("da", "Danish", "lix", lingua="DANISH"),
    Language("nb", "Norwegian (Bokmål)", "lix", lingua="BOKMAL"),
    Language("fi", "Finnish", "lix", lingua="FINNISH"),
    Language("ar", "Arabic", "osman", lingua="ARABIC"),
    Language("zh", "Chinese", None, word_based=False, lingua="CHINESE"),
    Language("ja", "Japanese", None, word_based=False, lingua="JAPANESE"),
    Language("xx", "Other (language-neutral metrics only)", "lix"),
]}

OTHER = "xx"

_LANG_ALIASES = {"no": "nb", "nn": "nb", "iw": "he", "jp": "ja", "cn": "zh"}

def normalize_lang_code(value: Any) -> Optional[str]:
    """Map 'es', 'es-MX', 'es_ES', 'Spanish' or 'spanish' to a code in LANGUAGES."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s or s == "nan":
        return None
    root = re.split(r"[-_]", s)[0]
    root = _LANG_ALIASES.get(root, root)
    if root in LANGUAGES:
        return root
    for code, lang in LANGUAGES.items():
        if lang.name.lower().split(" (")[0] == s:
            return code
    return None

def has_syllable_support(lang: str) -> bool:
    return lang != OTHER and pyphen.language_fallback(lang) is not None

def english_dictionary_ok() -> bool:
    """textstat counts English syllables with NLTK's cmudict, downloaded on first use.
    If the download is blocked (offline / proxy), English syllable metrics fail."""
    try:
        import nltk
        nltk.data.find("corpora/cmudict")
        return True
    except Exception:
        return False

@lru_cache(maxsize=None)
def get_engine(lang: str) -> textstatistics:
    """One textstat instance per language. textstat.set_lang() on the module is
    process-wide, so concurrent Streamlit sessions would overwrite each other."""
    eng = textstatistics()
    eng.set_lang(lang)
    return eng

# ----------------- Auto-detection (optional dependency: lingua) -----------------
# The detector knows every lingua language, not just the supported ones, so text in
# e.g. Croatian is reported as Croatian (scored with the neutral profile) instead of
# being forced into the closest supported language.
@lru_cache(maxsize=1)
def _detector():
    try:
        from lingua import LanguageDetectorBuilder
    except ImportError:
        return None
    return LanguageDetectorBuilder.from_all_languages().build()

def detection_available() -> bool:
    return _detector() is not None

_LINGUA_TO_CODE = {l.lingua: l.code for l in LANGUAGES.values() if l.lingua}
_LINGUA_TO_CODE["NYNORSK"] = "nb"

@dataclass(frozen=True)
class Detection:
    code: Optional[str]   # code in LANGUAGES, or None when not confident
    name: str             # detected language name, also for unsupported ones
    confidence: float

def detect_language(text: str, min_confidence: float = 0.2, min_lead: float = 1.5) -> Detection:
    """Detect the language of `text`. Accepts the top guess only if its confidence is
    at least `min_confidence` and `min_lead` times the runner-up's (or the runner-up
    maps to the same profile). Short fragments like 'OK' or 'Hola' are rejected."""
    det = _detector()
    if det is None or not text.strip():
        return Detection(None, "", 0.0)
    values = det.compute_language_confidence_values(text)
    if not values:
        return Detection(None, "", 0.0)
    top = values[0]
    code = _LINGUA_TO_CODE.get(top.language.name, OTHER)
    name = top.language.name.title()
    conf = float(top.value)
    second = values[1] if len(values) > 1 else None
    same_profile = second is not None and _LINGUA_TO_CODE.get(second.language.name, OTHER) == code
    leads = second is None or second.value == 0 or top.value / second.value >= min_lead
    if conf < min_confidence or not (leads or same_profile):
        return Detection(None, name, conf)
    return Detection(code, name, conf)

# ----------------- Text preparation -----------------
# Sentence terminators from other scripts, mapped to ones textstat recognises.
_TERMINATORS = {
    "。": ". ", "．": ". ", "！": "! ", "？": "? ",   # CJK full-width
    "؟": "? ", "۔": ". ",                             # Arabic / Urdu
    "।": ". ", "॥": ". ",                             # Devanagari danda
    "።": ". ", ";": "? ",                        # Ethiopic, Greek question mark
    "…": "...",
}
_TERM_RE = re.compile("|".join(map(re.escape, _TERMINATORS)))

_BULLET_RE = re.compile(r"^\s*(?:[*+\-•▪◦·‣–]|\d{1,3}[.)]|[a-zA-Z][.)])\s+")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+")
_ENDS_TERMINAL_RE = re.compile(r"[.!?。！？؟।…][\s*_\"'”’»)\]]*$")

def normalize_terminators(s: str) -> str:
    return _TERM_RE.sub(lambda m: _TERMINATORS[m.group(0)], s)

def strip_markdown_text(s: str) -> str:
    s = re.sub(r"(?m)^\s*#{1,6}\s+", "", s)               # # headings
    s = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", s)    # [text](url) -> text
    s = re.sub(r"`+([^`]*)`+", r"\1", s)                 # `code` -> code
    s = re.sub(r"(?m)^(\s*)[*+](\s+)", r"\1•\2", s)      # keep * / + bullets as •
    s = re.sub(r"\*+", "", s)                            # **bold**, *italics*
    s = re.sub(r"(?<!\w)_+|_+(?!\w)", "", s)             # __bold__, _italics_ (keeps snake_case)
    return s

def clean_text(s: Any, strip_markdown: bool) -> str:
    s = str(s)
    if strip_markdown:
        s = strip_markdown_text(s)
    # Spaced so "word—word" stays two words; an unspaced "-" would be deleted with
    # the punctuation and glue them into one long word.
    s = re.sub(r"\s*[—–]\s*", " - ", s)
    s = normalize_terminators(s)
    s = re.sub(r"[ \t ]+\n", "\n", s)
    s = re.sub(r"\n\s+", "\n", s)
    return s.strip()

def normalize_structure(text: str) -> str:
    """Give bullet, numbered-list and heading lines terminal punctuation so each
    counts as its own sentence instead of merging with the next line."""
    out = []
    for ln in str(text).splitlines():
        s = ln.rstrip()
        if (_BULLET_RE.match(s) or _HEADING_RE.match(s)) and not _ENDS_TERMINAL_RE.search(s):
            s += "."
        out.append(s)
    return "\n".join(out)

def prepare_variants(raw: Any, strip_markdown: bool) -> Dict[str, str]:
    # Structure is normalised on the raw text, before markdown stripping removes
    # the '*' / '#' markers it needs to see.
    return {
        "unprocessed": clean_text(raw, strip_markdown),
        "processed": clean_text(normalize_structure(raw), strip_markdown),
    }

# ----------------- Counts -----------------
_SENTENCE_RE = re.compile(r"\b[^.!?]+[.!?]*", re.UNICODE)

@dataclass(frozen=True)
class Counts:
    words: int
    sentences: int
    letters: int                  # alphabetic characters
    alnum: int                    # letters + digits (ARI "characters")
    chars_no_spaces: int
    chars_with_spaces: int
    long_words: int               # > 6 letters (LIX, RIX, WSTF)
    miniwords: int                # <= 3 letters (McAlpine EFLAW)
    syl: Optional[Tuple[int, ...]]  # syllables per word; None without a dictionary
    word_lengths: Tuple[int, ...]

    @property
    def syllables(self) -> int: return sum(self.syl)
    def at_least(self, n: int) -> int: return sum(1 for s in self.syl if s >= n)
    @property
    def asl(self) -> float: return self.words / self.sentences      # words per sentence
    @property
    def asw(self) -> float: return self.syllables / self.words      # syllables per word

def count_sentences(text: str, ignore_short: bool) -> int:
    """textstat's splitter. With ignore_short (textstat's own behaviour) sentences of
    ≤2 words are not counted — though their words still are."""
    if not text:
        return 0
    parts = _SENTENCE_RE.findall(text)
    if ignore_short:
        return max(1, len(parts) - sum(1 for p in parts if len(_words(p)) <= 2))
    return max(1, len(parts))

def _words(text: str) -> List[str]:
    # Same tokenisation as textstat.lexicon_count(removepunct=True)
    return re.sub(r"[^\w\s]", "", text).split()

@lru_cache(maxsize=20000)
def get_counts(text: str, lang: str, ignore_short: bool) -> Counts:
    eng = get_engine(lang)
    words = _words(text)
    syl = tuple(eng.syllable_count(w) for w in words) if has_syllable_support(lang) else None
    return Counts(
        words=len(words),
        sentences=count_sentences(text, ignore_short),
        letters=sum(c.isalpha() for c in text),
        alnum=sum(c.isalnum() for c in text),
        chars_no_spaces=len(re.sub(r"\s", "", text)),
        chars_with_spaces=len(text),
        long_words=sum(1 for w in words if len(w) > 6),
        miniwords=sum(1 for w in words if len(w) <= 3),
        syl=syl,
        word_lengths=tuple(len(re.sub(r"\d", "", w)) for w in words if re.sub(r"\d", "", w)),
    )

# ----------------- Formulas -----------------
# Each takes (counts, text, lang, engine). Language-dependent coefficients sit next
# to the formula they belong to.

FLESCH = {  # base, × words per sentence, × syllables per word
    "en": (206.835, 1.015, 84.6),   # Flesch 1948
    "de": (180.0, 1.0, 58.5),       # Amstad 1978
    "fr": (207.0, 1.015, 73.6),     # Kandel & Moles 1958
    "it": (217.0, 1.3, 60.0),       # Franchina & Vacca 1986
    "nl": (206.84, 0.93, 77.0),     # Douma 1960
    "pt": (248.835, 1.015, 84.6),   # Martins et al. 1996
    "ru": (206.835, 1.3, 60.1),     # Oborneva 2006
}
FKGL = {  # × words per sentence, × syllables per word, constant
    "en": (0.39, 11.8, -15.59),     # Kincaid et al. 1975
    "ru": (0.5, 8.4, -15.59),       # Oborneva 2006
}

def _flesch(c: Counts, t: str, lang: str, e) -> float:
    base, a, b = FLESCH.get(lang, FLESCH["en"])
    return base - a * c.asl - b * c.asw

def _fkgl(c: Counts, t: str, lang: str, e) -> float:
    a, b, k = FKGL.get(lang, FKGL["en"])
    return a * c.asl + b * c.asw + k

def _smog(c, t, lang, e):
    return 1.0430 * math.sqrt(c.at_least(3) * 30 / c.sentences) + 3.1291

def _coleman_liau(c, t, lang, e):
    # textstat uses 0.058; Coleman & Liau (1975) give 0.0588
    return 0.0588 * (100 * c.letters / c.words) - 0.296 * (100 * c.sentences / c.words) - 15.8

def _ari(c, t, lang, e):
    # "characters" = letters and digits; textstat also counts punctuation
    return 4.71 * (c.alnum / c.words) + 0.5 * c.asl - 21.43

def _fog_complex_words(c: Counts, t: str, e) -> int:
    """Gunning's rules: 3+ syllables, but not proper nouns, hyphenated compounds,
    or words that only reach 3 syllables through -es / -ed / -ing."""
    n, sentence_start = 0, True
    for tok in t.split():
        word = re.sub(r"[^\w'-]", "", tok)
        if word and not any(ch.isdigit() for ch in word) and "-" not in word \
                and (sentence_start or not word[0].isupper()):
            clean = re.sub(r"[^\w]", "", word)
            syl = e.syllable_count(clean)
            m = re.search(r"(es|ed|ing)$", clean.lower())
            if syl >= 3 and m and e.syllable_count(clean[: -len(m.group(1))]) < syl:
                syl -= 1
            n += syl >= 3
        sentence_start = bool(re.search(r"[.!?][\"')\]]*$", tok))
    return n

def _gunning_fog(c, t, lang, e):
    return 0.4 * (c.asl + 100 * _fog_complex_words(c, t, e) / c.words)

def _fog_pl(c, t, lang, e):
    # FOG-PL: Gunning Fog with "hard" = 4+ syllables (Polish words are longer)
    return 0.4 * (c.asl + 100 * c.at_least(4) / c.words)

def _pisarek(c, t, lang, e):
    # Pisarek (1969/2007), non-linear form used by Jasnopis: √(ASL² + %hard²) / 2
    return math.sqrt(c.asl ** 2 + (100 * c.at_least(4) / c.words) ** 2) / 2

def _dale_chall(c, t, lang, e):
    pdw = 100 * sum(1 for w in _words(t) if e.is_difficult_word(w, 0)) / c.words
    return 0.1579 * pdw + 0.0496 * c.asl + (3.6365 if pdw > 5 else 0)

def _spache(c, t, lang, e):
    # Original 1953 coefficients; textstat's Dale-Chall list stands in for Spache's
    # own list and 1-syllable words always count as easy — an approximation.
    pdw = 100 * sum(1 for w in _words(t) if e.is_difficult_word(w, 2)) / c.words
    return 0.141 * c.asl + 0.086 * pdw + 0.839

def _linsear(c, t, lang, e):
    easy = sum(1 for s in c.syl if 0 < s <= 2)
    hard = c.at_least(3)
    r = (easy + 3 * hard) / c.sentences
    return r / 2 if r > 20 else (r - 2) / 2

def _eflaw(c, t, lang, e):
    return (c.words + c.miniwords) / c.sentences

def _fernandez_huerta(c, t, lang, e):
    # As corrected by Law (2011): the usually printed "sentences per 100 words" term
    # makes more sentences *lower* the score; it has to be words per sentence.
    return 206.84 - 60 * c.asw - 1.02 * c.asl

def _szigriszt_pazos(c, t, lang, e):
    return 206.835 - 62.3 * c.asw - c.asl

def _gutierrez_polini(c, t, lang, e):
    return 95.2 - 9.7 * (c.letters / c.words) - 0.35 * c.asl

def _crawford(c, t, lang, e):
    return -0.205 * (100 * c.sentences / c.words) + 0.049 * (100 * c.syllables / c.words) - 3.407

def _legibilidad_mu(c, t, lang, e):
    # Muñoz Baquedano & Muñoz Urra (2006): µ = n/(n−1) · x̄/σ² · 100, letters per word,
    # σ² the population variance
    n = len(c.word_lengths)
    var = statistics.pvariance(c.word_lengths) if n > 1 else 0
    return (n / (n - 1)) * (statistics.mean(c.word_lengths) / var) * 100 if var else None

def _flesch_vacca(c, t, lang, e):
    # Franchina & Vacca 1972: 206 − 0.65 × syllables per 100 words − words per sentence
    return 206 - 0.65 * 100 * c.asw - c.asl

def _gulpease(c, t, lang, e):
    # Lucisano & Piemontese 1988; letters, where textstat counts all characters
    return 89 + (300 * c.sentences - 10 * c.letters) / c.words

def _brouwer(c, t, lang, e):
    # Brouwer 1963: 195 − 2 × words per sentence − 0.67 × syllables per 100 words
    return 195 - 2 * c.asl - 67 * c.asw

def _wstf1(c, t, lang, e):
    ms = 100 * c.at_least(3) / c.words
    iw = 100 * c.long_words / c.words
    es = 100 * sum(1 for s in c.syl if s == 1) / c.words
    return 0.1935 * ms + 0.1672 * c.asl + 0.1297 * iw - 0.0327 * es - 0.875

def _lix(c, t, lang, e):
    return c.asl + 100 * c.long_words / c.words

def _rix(c, t, lang, e):
    return c.long_words / c.sentences

def _osman(c, t, lang, e):
    return e.osman(t)  # textstat's implementation; see FORMULAS.md for caveats

def _cjk_sentences(t: str) -> int:
    return len([p for p in re.split(r"[.!?]+", t) if p.strip()])

def _cjk_chars(t: str) -> int:
    return len(re.findall(r"\w", t))

# ----------------- Metric registry -----------------
@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    fn: Callable[..., Any]
    langs: Optional[frozenset] = None  # None = every word-based language
    syllables: bool = False            # needs a syllable dictionary (pyphen)
    legacy: bool = False               # part of the original en/es column set
    scale: Optional[str] = None        # key into SCALES for a difficulty level
    formula: bool = True               # False for plain counts
    word_based: bool = True            # False = for zh/ja only

def _l(*codes: str) -> frozenset:
    return frozenset(codes)

EN = _l("en")
ES = _l("es")
METRICS: List[Metric] = [
    Metric("smog_index", "SMOG", _smog, EN, True, True, "grade"),
    Metric("flesch_reading_ease", "Flesch Reading Ease (language-adapted)", _flesch,
           frozenset(FLESCH), True, True, "flesch"),
    Metric("flesch_kincaid_grade", "Flesch-Kincaid Grade (language-adapted)", _fkgl,
           frozenset(FKGL), True, True, "grade"),
    Metric("gunning_fog", "Gunning Fog", _gunning_fog, EN, True, True, "grade"),
    Metric("coleman_liau_index", "Coleman-Liau", _coleman_liau, EN, False, True, "grade"),
    Metric("automated_readability_index", "Automated Readability Index", _ari, EN, False, True, "grade"),
    Metric("dale_chall_readability_score", "Dale-Chall", _dale_chall, EN, False, True, "dale_chall"),
    # 🇪🇸 Spanish
    Metric("fernandez_huerta", "Fernández Huerta", _fernandez_huerta, ES, True, True, "fernandez_huerta"),
    Metric("szigriszt_pazos", "Szigriszt-Pazos / INFLESZ", _szigriszt_pazos, ES, True, True, "inflesz"),
    Metric("gutierrez_polini", "Gutiérrez de Polini", _gutierrez_polini, ES, False, True, "gutierrez"),
    Metric("crawford", "Crawford (years of schooling)", _crawford, ES, True, True, "grade"),
    Metric("legibilidad_mu", "Legibilidad µ", _legibilidad_mu, ES, False, False, "mu"),
    Metric("linsear_write_formula", "Linsear Write", _linsear, EN, True, True, "grade"),
    Metric("spache_readability", "Spache (approximate)", _spache, EN, True, True, "grade"),
    Metric("mcalpine_eflaw", "McAlpine EFLAW", _eflaw, EN, False, False, "eflaw"),
    # Other languages
    Metric("flesch_vacca", "Flesch-Vacca 1972", _flesch_vacca, _l("it"), True, False, "flesch"),
    Metric("gulpease_index", "Gulpease", _gulpease, _l("it"), False, False, "gulpease"),
    Metric("brouwer", "Brouwer Leesindex", _brouwer, _l("nl"), True, False, "flesch"),
    Metric("wiener_sachtextformel", "Wiener Sachtextformel 1", _wstf1, _l("de"), True, False, "grade"),
    Metric("fog_pl", "FOG-PL", _fog_pl, _l("pl"), True, False, "pl_school"),
    Metric("pisarek", "Pisarek", _pisarek, _l("pl"), True, False, "pl_school"),
    Metric("osman", "OSMAN", _osman, _l("ar")),
    # Language-neutral (word/letter based)
    Metric("lix", "LIX", _lix, scale="lix"),
    Metric("rix", "RIX", _rix, scale="rix"),
    # Counts
    Metric("sentences", "Sentences", lambda c, t, l, e: c.sentences, legacy=True, formula=False),
    Metric("words", "Words", lambda c, t, l, e: c.words, legacy=True, formula=False),
    Metric("syllables", "Syllables", lambda c, t, l, e: c.syllables, syllables=True, legacy=True, formula=False),
    Metric("polysyllables", "Words with 3+ syllables", lambda c, t, l, e: c.at_least(3),
           syllables=True, legacy=True, formula=False),
    Metric("monosyllables", "Words with 1 syllable", lambda c, t, l, e: sum(1 for s in c.syl if s == 1),
           syllables=True, legacy=True, formula=False),
    Metric("long_words", "Words with 7+ letters", lambda c, t, l, e: c.long_words, formula=False),
    Metric("characters_no_spaces", "Characters (no spaces)", lambda c, t, l, e: c.chars_no_spaces,
           legacy=True, formula=False),
    Metric("characters_with_spaces", "Characters (with spaces)", lambda c, t, l, e: c.chars_with_spaces,
           legacy=True, formula=False),
    Metric("avg_words_per_sentence", "Words per sentence", lambda c, t, l, e: c.asl, formula=False),
    Metric("avg_syllables_per_word", "Syllables per word", lambda c, t, l, e: c.asw,
           syllables=True, legacy=True, formula=False),
    # Scripts without word spacing (zh, ja)
    Metric("cjk_sentences", "Sentences", lambda c, t, l, e: _cjk_sentences(t), formula=False, word_based=False),
    Metric("cjk_characters", "Characters", lambda c, t, l, e: _cjk_chars(t), formula=False, word_based=False),
    Metric("cjk_chars_per_sentence", "Characters per sentence",
           lambda c, t, l, e: _cjk_chars(t) / max(1, _cjk_sentences(t)), formula=False, word_based=False),
]
# Source and audit status for each formula (details in FORMULAS.md)
REFERENCES: Dict[str, Tuple[str, str]] = {
    "smog_index": ("McLaughlin 1969", "✓ matches source"),
    "flesch_reading_ease": ("Flesch 1948; de Amstad 1978; fr Kandel & Moles 1958; it Franchina & Vacca 1986; "
                            "nl Douma 1960; pt Martins et al. 1996; ru Oborneva 2006", "✓ matches source"),
    "flesch_kincaid_grade": ("Kincaid et al. 1975; ru Oborneva 2006", "✓ — Russian now uses Oborneva's coefficients"),
    "gunning_fog": ("Gunning 1952", "fixed — textstat excluded every word on the Dale-Chall easy list"),
    "coleman_liau_index": ("Coleman & Liau 1975", "fixed — textstat used 0.058 instead of 0.0588"),
    "automated_readability_index": ("Smith & Senter 1967", "fixed — textstat counted punctuation as characters"),
    "dale_chall_readability_score": ("Dale & Chall 1948", "✓ matches source"),
    "fernandez_huerta": ("Fernández Huerta 1959, corrected by Law 2011", "✓ corrected form"),
    "szigriszt_pazos": ("Szigriszt-Pazos 1993; INFLESZ scale Barrio-Cantalejo 2008", "✓ matches source"),
    "gutierrez_polini": ("Gutiérrez de Polini 1972", "✓ formula; designed for 6th-grade texts, scale unverified"),
    "crawford": ("Crawford 1984", "✓ result is years of primary schooling"),
    "legibilidad_mu": ("Muñoz Baquedano & Muñoz Urra 2006", "new"),
    "linsear_write_formula": ("O'Hayre 1966", "✓ applied to the whole text, not a 100-word sample"),
    "spache_readability": ("Spache 1953", "⚠ approximate: uses the Dale-Chall list, not Spache's"),
    "mcalpine_eflaw": ("McAlpine 2006", "✓ matches source"),
    "flesch_vacca": ("Franchina & Vacca 1972", "new"),
    "gulpease_index": ("Lucisano & Piemontese 1988", "fixed — textstat counted all characters, not letters"),
    "brouwer": ("Brouwer 1963", "new"),
    "wiener_sachtextformel": ("Bamberger & Vanecek 1984", "✓ matches source (variant 1, prose)"),
    "fog_pl": ("Gunning Fog with 4+ syllable hard words (Broda et al. 2014)", "✓"),
    "pisarek": ("Pisarek 1969/2007, as used by Jasnopis", "new; hard words should be counted on lemmas"),
    "osman": ("El-Haj & Rayson 2016", "⚠ textstat's version; syllables need diacritics"),
    "lix": ("Björnsson 1968", "✓ matches source"),
    "rix": ("Anderson 1983", "✓ matches source"),
}

METRICS_BY_KEY = {m.key: m for m in METRICS}
METRIC_ORDER = [m.key for m in METRICS]

def is_native(m: Metric, lang: str) -> bool:
    """Calibrated for this language (or language-neutral), as opposed to a legacy extra."""
    return m.langs is None or lang in m.langs

def applicable_metrics(lang: str, include_legacy: bool = False) -> List[Metric]:
    info = LANGUAGES[lang]
    syl = has_syllable_support(lang)
    out = []
    for m in METRICS:
        if m.word_based != info.word_based or (m.syllables and not syl):
            continue
        if is_native(m, lang) or (include_legacy and m.legacy):
            out.append(m)
    return out

def formula_options(lang: str) -> List[str]:
    """Metric keys that can serve as the headline for `lang`."""
    return [m.key for m in applicable_metrics(lang) if m.formula]

# ----------------- Difficulty levels -----------------
# Each row: (threshold, level 1–5 where 1 = easiest, label). "higher_is_easier"
# scales take the first row whose threshold the score reaches; the others the
# first row whose threshold the score does not exceed.
_INF = float("inf")
SCALES: Dict[str, Dict[str, Any]] = {
    "grade": {"higher_is_easier": False, "rows": [
        (6, 1, "Very easy (grade ≤6)"), (8, 2, "Easy / plain language (grade 7–8)"),
        (10, 3, "Standard (grade 9–10)"), (13, 4, "Difficult (grade 11–13)"),
        (_INF, 5, "Very difficult (college+)")]},
    "flesch": {"higher_is_easier": True, "rows": [
        (90, 1, "Very easy"), (80, 1, "Easy"), (70, 2, "Fairly easy"), (60, 3, "Standard"),
        (50, 4, "Fairly difficult"), (30, 4, "Difficult"), (-_INF, 5, "Very difficult")]},
    "fernandez_huerta": {"higher_is_easier": True, "rows": [
        (90, 1, "Muy fácil"), (80, 1, "Fácil"), (70, 2, "Bastante fácil"), (60, 3, "Normal"),
        (50, 4, "Bastante difícil"), (30, 4, "Difícil"), (-_INF, 5, "Muy difícil")]},
    "inflesz": {"higher_is_easier": True, "rows": [  # Barrio-Cantalejo et al. 2008
        (80, 1, "Muy fácil"), (65, 2, "Bastante fácil"), (55, 3, "Normal"),
        (40, 4, "Algo difícil"), (-_INF, 5, "Muy difícil")]},
    "gutierrez": {"higher_is_easier": True, "rows": [
        (66.66, 2, "Fácil"), (33.33, 3, "Normal"), (-_INF, 4, "Difícil")]},
    "mu": {"higher_is_easier": True, "rows": [  # Muñoz Baquedano, user manual table 1
        (91, 1, "Muy fácil"), (81, 1, "Fácil"), (71, 2, "Un poco fácil"), (61, 3, "Adecuado"),
        (51, 4, "Un poco difícil"), (31, 4, "Difícil"), (-_INF, 5, "Muy difícil")]},
    "gulpease": {"higher_is_easier": True, "rows": [
        (80, 1, "Easy for primary-school readers"), (60, 2, "Easy for middle-school readers"),
        (40, 3, "Easy for high-school readers"), (-_INF, 5, "Very difficult")]},
    "dale_chall": {"higher_is_easier": False, "rows": [
        (4.9, 1, "Grade 4 and below"), (5.9, 1, "Grades 5–6"), (6.9, 2, "Grades 7–8"),
        (7.9, 3, "Grades 9–10"), (8.9, 4, "Grades 11–12"), (_INF, 5, "College")]},
    "eflaw": {"higher_is_easier": False, "rows": [
        (20, 1, "Very easy"), (25, 2, "Quite easy"), (29, 3, "Mildly difficult"),
        (_INF, 5, "Very confusing")]},
    "lix": {"higher_is_easier": False, "rows": [
        (30, 1, "Very easy"), (40, 2, "Easy"), (50, 3, "Medium"),
        (60, 4, "Difficult"), (_INF, 5, "Very difficult")]},
    "rix": {"higher_is_easier": False, "rows": [  # Anderson 1983 grade table
        (1.8, 1, "Grade ≤6"), (3.0, 2, "Grades 7–8"), (4.5, 3, "Grades 9–10"),
        (7.2, 4, "Grades 11–12"), (_INF, 5, "College")]},
    "pl_school": {"higher_is_easier": False, "rows": [
        (6, 1, "Bardzo łatwy (szkoła podstawowa)"), (9, 2, "Łatwy"), (12, 3, "Średnio trudny (liceum)"),
        (15, 4, "Trudny (studia)"), (_INF, 5, "Bardzo trudny")]},
}

def difficulty_band(scale: Optional[str], score: Optional[float]) -> Tuple[Optional[int], Optional[str]]:
    if scale is None or score is None:
        return None, None
    spec = SCALES[scale]
    for threshold, level, label in spec["rows"]:
        if (score >= threshold) if spec["higher_is_easier"] else (score <= threshold):
            return level, label
    return None, None

# ----------------- Scoring -----------------
def _num(v: Any) -> Any:
    return round(float(v), 3) if isinstance(v, float) else v

@lru_cache(maxsize=20000)
def _score_cached(raw: str, lang: str, strip_markdown: bool, include_legacy: bool,
                  headline: Optional[str], ignore_short: bool) -> Tuple[Tuple[str, Any], ...]:
    eng = get_engine(lang)
    metrics = applicable_metrics(lang, include_legacy)
    headline = headline if headline in formula_options(lang) else LANGUAGES[lang].headline
    out: Dict[str, Any] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # textstat warns when a language lacks an easy-word list
        for variant, text in prepare_variants(raw, strip_markdown).items():
            try:
                counts = get_counts(text, lang, ignore_short)
            except Exception:
                counts = None
            levels, neutral_levels = [], []
            for m in metrics:
                try:
                    v = _num(m.fn(counts, text, lang, eng)) if counts and counts.words else None
                except Exception:
                    v = None
                out[f"{m.key}_{variant}"] = v
                if m.scale and m.formula:
                    level, _ = difficulty_band(m.scale, v)
                    out[f"{m.key}_level_{variant}"] = level
                    if level is not None and m.langs and lang in m.langs:
                        levels.append(level)
                    elif level is not None and m.langs is None:
                        neutral_levels.append(level)
            score = out.get(f"{headline}_{variant}") if headline else None
            level, label = difficulty_band(METRICS_BY_KEY[headline].scale if headline else None, score)
            out[f"headline_score_{variant}"] = score
            out[f"difficulty_level_{variant}"] = level
            out[f"difficulty_{variant}"] = label
            # Median level over the formulas built for this language. LIX/RIX only count
            # where a language has nothing else: their thresholds come from Swedish and
            # overrate languages with long words (e.g. Polish).
            levels = levels or neutral_levels
            out[f"consensus_level_{variant}"] = statistics.median(levels) if levels else None
            out[f"formulas_counted_{variant}"] = len(levels)
    out["headline_metric"] = headline
    return tuple(out.items())

def score_text(raw: Any, lang: str, strip_markdown: bool = True, include_legacy: bool = False,
               headline: Optional[str] = None, ignore_short_sentences: bool = True) -> Dict[str, Any]:
    """Score one cell. `headline` overrides the language's default headline formula;
    `ignore_short_sentences` keeps textstat's rule of not counting ≤2-word sentences."""
    if lang not in LANGUAGES:
        raise ValueError(f"Unsupported language code: {lang!r}")
    return dict(_score_cached(str(raw), lang, strip_markdown, include_legacy, headline, ignore_short_sentences))

HEADLINE_COLS = ["headline_metric",
                 "headline_score_unprocessed", "difficulty_level_unprocessed", "difficulty_unprocessed",
                 "consensus_level_unprocessed", "formulas_counted_unprocessed",
                 "headline_score_processed", "difficulty_level_processed", "difficulty_processed",
                 "consensus_level_processed", "formulas_counted_processed"]

def ordered_columns(present: List[str], leading: List[str]) -> List[str]:
    """Leading id columns, headline columns, then every metric present —
    all *_unprocessed first, then all *_processed, in METRIC_ORDER, each formula
    followed by its difficulty level."""
    cols = [c for c in leading if c in present] + [c for c in HEADLINE_COLS if c in present]
    for variant in ("unprocessed", "processed"):
        for k in METRIC_ORDER:
            cols += [c for c in (f"{k}_{variant}", f"{k}_level_{variant}") if c in present]
    return cols
