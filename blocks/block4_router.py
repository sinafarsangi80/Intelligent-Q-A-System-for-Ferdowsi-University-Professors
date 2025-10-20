# -*- coding: utf-8 -*-
"""
Block 4 — Router helpers used by server.py

Exports used by server.py:
  - normalize_text(str) -> str
  - extract_track_a(str) -> (intent|None, name|None)
      intent in {"email", "homepage", "pubs"} when and only when the query
      is professor-specific. Otherwise returns (None, None) so server routes to Track B.

Design:
  - If the query mentions topic heads (حوزه/زمینه/رشته/گرایش) or common research
    keywords and does NOT include an explicit professor hint (استاد/دکتر/پروفسور),
    we DO NOT mark it as Track A.
  - Negative lookahead prevents capturing names that start with "حوزه|زمینه|رشته|گرایش".
"""

import re
from typing import Optional, Tuple

# -------------------------
# Persian normalization (lightweight)
# -------------------------
DIACRITICS_RE   = re.compile("[" "\u064B-\u065F" "\u0670" "\u06D6-\u06ED" "]")
TATWEEL_RE      = re.compile("\u0640")
NBSP_RE         = re.compile("[\u00A0\u2007\u202F]")
ZWS_RE          = re.compile("[\u200B\u200E\u200F]")
ZWNJ_RE         = re.compile("\u200C+")
MULTI_SPACE_RE  = re.compile(r"\s+")
ZWNJ            = "\u200C"

def _unify_arabic_persian_letters(s: str) -> str:
    if not isinstance(s, str):
        return s
    return s.replace("\u064A", "\u06CC").replace("\u0643", "\u06A9")  # ي→ی, ك→ک

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = _unify_arabic_persian_letters(s)
    s = DIACRITICS_RE.sub("", s)
    s = TATWEEL_RE.sub("", s)
    s = ZWS_RE.sub("", s)
    s = ZWNJ_RE.sub(ZWNJ, s)
    s = re.sub(r"\s*"+ZWNJ+r"\s*", ZWNJ, s)
    s = NBSP_RE.sub(" ", s)
    s = MULTI_SPACE_RE.sub(" ", s).strip()
    return s

# -------------------------
# Intent parsing config
# -------------------------
# Professor hints => allow Track A even if keywords appear
PROF_HINT_RE = re.compile(r"(?:\b|^)(استاد|دکتر|پروفسور|هیئت\s*علمی)(?:\b|$)")

# Topic heads and some frequent keywords; extend as your corpus grows
TOPIC_HEAD_RE = re.compile(r"\b(?:حوزه(?:ی)?|زمینه(?:ی)?|رشته(?:ی)?|گرایش)\b")
TOPIC_KEYWORDS = [
    "پردازش تصویر", "بینایی ماشین", "یادگیری ماشین", "داده کاوی",
    "هوش مصنوعی", "شبکه های عصبی", "شبکه‌های عصبی", "پردازش زبان طبیعی",
    "رایانش ابری", "بهینه سازی", "بهینه‌سازی", "سیگنال", "رباتیک", "معماری نرم‌افزار"
]
TOPIC_KW_RE = re.compile("|".join(map(re.escape, TOPIC_KEYWORDS)))

# Base patterns for A-intents (email/homepage/pubs)
# NOTE: P_A_PUBS uses negative-lookahead to avoid names that start with topic heads.
P_A_EMAIL = re.compile(
    r"(?:^|\s)(?:ایمیل)\s+(?:استاد\s+)?(?P<name>[\w\u0600-\u06FF\u200c\s]{2,})"
)
P_A_HOME = re.compile(
    r"(?:^|\s)(?:صفحه\s*شخصی|وب[‌\s]*سایت)\s+(?:استاد\s+)?(?P<name>[\w\u0600-\u06FF\u200c\s]{2,})"
)
P_A_PUBS = re.compile(
    r"(?:^|\s)(?:لیست\s+مقالات|مقالات)\s+(?:استاد\s+)?(?P<name>(?!(?:حوزه|زمینه|رشته|گرایش)\b)[\w\u0600-\u06FF\u200c\s]{2,})"
)

# -------------------------
# Utilities
# -------------------------
def _looks_like_topic(q: str) -> bool:
    """True if the query is likely topic/field oriented."""
    if TOPIC_HEAD_RE.search(q):
        return True
    if TOPIC_KW_RE.search(q):
        return True
    return False

def _has_prof_hint(q: str) -> bool:
    return bool(PROF_HINT_RE.search(q))

def _clean_name(nm: str) -> str:
    nm = nm.strip(" ؟?!.،,:؛")
    nm = re.sub(r"(?:چیست|کجاست|لطفاً?|لطفا)$", "", nm).strip()
    return nm

# -------------------------
# Public API
# -------------------------
def extract_track_a(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (intent, name) where intent in {"email","homepage","pubs"} for professor-specific queries.
    Otherwise returns (None, None) so server routes to Track B.

    Examples:
      "مقالات استاد احمد آفتابی ثانی"  -> ("pubs", "احمد آفتابی ثانی")
      "مقالات حوزه پردازش تصویر"       -> (None, None)   # topic => Track B
    """
    q = normalize_text(query)

    # If it looks like a topic/field query and there's no explicit professor hint, route to B.
    if _looks_like_topic(q) and not _has_prof_hint(q):
        return (None, None)

    # Try specific intents (A) in order
    for intent, pat in (("email", P_A_EMAIL), ("homepage", P_A_HOME), ("pubs", P_A_PUBS)):
        m = pat.search(q)
        if not m:
            continue
        nm = _clean_name(m.group("name"))
        # Guard: if the captured "name" is actually a topic phrase (e.g., "حوزه پردازش تصویر"), route to B.
        if nm and _looks_like_topic(nm) and not _has_prof_hint(q):
            return (None, None)
        return (intent, nm if nm else None)

    # If an A-intent word is present but name missing → let server ask for clarification (“clarify” route)
    if re.search(r"\bایمیل\b", q):
        return ("email", None)
    if re.search(r"(?:صفحه\s*شخصی|وب[‌\s]*سایت)", q):
        return ("homepage", None)
    if re.search(r"(?:لیست\s+مقالات|مقالات)\b", q):
        # still ensure not purely topic-style if no professor hint
        if _looks_like_topic(q) and not _has_prof_hint(q):
            return (None, None)
        return ("pubs", None)

    # No A-intent detected
    return (None, None)
