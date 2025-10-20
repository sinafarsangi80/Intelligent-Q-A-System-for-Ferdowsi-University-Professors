# -*- coding: utf-8 -*-
"""
Block 3 — Deterministic profile lookup (Track A) (Local)

Usage examples:
  # full profile summary (email, homepage, top-5 publications)
  python blocks/block3_lookup.py --data data/normalized/records.jsonl --name "احمد آفتابی ثانی"

  # only publications (top-5)
  python blocks/block3_lookup.py --data data/normalized/records.jsonl --name "احمد آفتابی ثانی" --top 5 --only pubs

  # only email/homepage
  python blocks/block3_lookup.py --data data/normalized/records.jsonl --name "احمد آفتابی ثانی" --only contacts

Notes:
- Deterministic: exact match on normalized Persian name fields (نام + نام خانوادگی).
- Optional: minor-typo tolerance if you pass --fuzzy (Persian only).
- Sort rule for publications: keep original order but prioritize those with a usable 'article_link'.
"""

import argparse, json, os, re, sys, time
from typing import Any, Dict, List, Optional, Iterable, Tuple

try:
    from pydantic import BaseModel, Field
except Exception:
    BaseModel = object
    def Field(default=None, **kwargs): return default

# Optional minor-typo matching
try:
    from rapidfuzz import process, fuzz
    HAVE_FUZZ = True
except Exception:
    HAVE_FUZZ = False

# -------------------------
# Models
# -------------------------
class Publication(BaseModel):
    title: Optional[str] = Field(default=None)
    authors: Optional[List[str]] = Field(default=None)
    journal: Optional[str] = Field(default=None)
    article_link: Optional[str] = Field(default=None)
    year: Optional[int] = Field(default=None)

class Professor(BaseModel):
    # Persian fields only for Track A
    نام: Optional[str] = Field(default=None)
    نام_خانوادگی: Optional[str] = Field(default=None, alias="نام خانوادگی")
    دانشکده: Optional[str] = Field(default=None)
    گروه_آموزشی: Optional[str] = Field(default=None, alias="گروه آموزشی")
    وضعیت_اشتغال: Optional[str] = Field(default=None, alias="وضعیت اشتغال")
    پست_الکترونیکی: Optional[str] = Field(default=None, alias="پست الکترونیکی")
    صفحه_شخصی: Optional[str] = Field(default=None, alias="صفحه شخصی")
    publications: Optional[List[Publication]] = Field(default=None)

    def full_name(self) -> str:
        # prefer "نام + نام خانوادگی"
        first = self.نام or ""
        last = (self.نام_خانوادگی if self.نام_خانوادگی is not None else getattr(self, "نام خانوادگی", "")) or ""
        first = str(first)
        last  = str(last)
        return (first + " " + last).strip()

# -------------------------
# Minimal normalizer (mirrors Block 2 rules relevant to names)
# -------------------------
DIACRITICS_RE = re.compile("[" "\u064B-\u065F" "\u0670" "\u06D6-\u06ED" "]")
TATWEEL_RE = re.compile("\u0640")
NBSP_RE = re.compile("[\u00A0\u2007\u202F]")
ZWS_RE = re.compile("[\u200B\u200E\u200F]")
ZWNJ_RE = re.compile("\u200C+")
MULTI_SPACE_RE = re.compile(r"\s+")
ZWNJ = "\u200C"

def unify_arabic_persian_letters(s: str) -> str:
    return s.replace("\u064A", "\u06CC").replace("\u0643", "\u06A9")  # ي→ی, ك→ک

def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unify_arabic_persian_letters(s)
    s = DIACRITICS_RE.sub("", s)
    s = TATWEEL_RE.sub("", s)
    s = ZWS_RE.sub("", s)
    s = ZWNJ_RE.sub(ZWNJ, s)
    s = re.sub(r"\s*"+ZWNJ+r"\s*", ZWNJ, s)
    s = NBSP_RE.sub(" ", s)
    s = MULTI_SPACE_RE.sub(" ", s).strip()
    return s

# -------------------------
# IO helpers
# -------------------------
def iter_jsonl_or_array(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        if head.lstrip().startswith("["):
            data = json.load(f)
            if isinstance(data, list):
                for r in data:
                    if isinstance(r, dict):
                        yield r
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if isinstance(r, dict):
                    yield r

# -------------------------
# Index
# -------------------------
class ProfIndex:
    def __init__(self):
        self.by_key: Dict[str, Professor] = {}
        self.keys: List[str] = []  # for fuzzy

    @staticmethod
    def _mk_keys(rec: Professor) -> List[str]:
        keys = []
        fn = normalize_name(rec.full_name())
        if fn:
            keys.append(fn)
        # Also allow "last first"
        first = normalize_name(rec.نام or "")
        last  = normalize_name(rec.نام_خانوادگی if rec.نام_خانوادگی is not None else getattr(rec, "نام خانوادگی", ""))
        lf = (last + " " + first).strip()
        if lf:
            keys.append(lf)
        return list(dict.fromkeys([k for k in keys if k]))  # unique, keep order

    def add(self, rec_raw: Dict[str, Any]) -> None:
        # validate to Pydantic (optional; if pydantic missing, still fine)
        try:
            rec = Professor(**rec_raw)
        except Exception:
            # Soft-fallback: wrap raw dict
            rec = Professor()
            for k, v in rec_raw.items():
                try:
                    setattr(rec, k.replace(" ", "_"), v)
                except Exception:
                    pass
            if "publications" in rec_raw:
                rec.publications = rec_raw["publications"]
        keys = self._mk_keys(rec)
        for k in keys:
            # First wins; deterministic (no ambiguity). If collision, keep the first loaded.
            if k not in self.by_key:
                self.by_key[k] = rec
        self.keys.extend(keys)

    def find_exact(self, q: str) -> Optional[Professor]:
        k = normalize_name(q)
        return self.by_key.get(k) or None

    def find_fuzzy(self, q: str, limit: int = 3, score_cutoff: int = 92) -> List[Tuple[str, int, Professor]]:
        if not HAVE_FUZZ:
            return []
        qn = normalize_name(q)
        # RapidFuzz returns (key, score, idx)
        matches = process.extract(
            qn, list(self.by_key.keys()),
            scorer=fuzz.WRatio, limit=limit, score_cutoff=score_cutoff
        )
        out: List[Tuple[str, int, Professor]] = []
        for key, score, _ in matches:
            out.append((key, score, self.by_key[key]))
        return out

# -------------------------
# Business logic
# -------------------------
def _pub_get(p: Any, key: str, default=None):
    # Read a field from dict or Pydantic model (v2: model_dump, v1: dict/attr).
    if isinstance(p, dict):
        return p.get(key, default)
    # Pydantic v2
    if hasattr(p, "model_dump"):
        return p.model_dump(exclude_none=True).get(key, default)
    # Pydantic v1
    if hasattr(p, "dict"):
        return p.dict(exclude_none=True).get(key, default)
    # Fallback to attribute
    return getattr(p, key, default)

def _pub_to_dict(p: Any) -> Dict[str, Any]:
    if isinstance(p, dict):
        return p
    if hasattr(p, "model_dump"):       # Pydantic v2
        return p.model_dump(exclude_none=True)
    if hasattr(p, "dict"):             # Pydantic v1
        return p.dict(exclude_none=True)
    # Best-effort fallback
    out = {}
    for k in ("title", "authors", "journal", "article_link", "year"):
        out[k] = getattr(p, k, None)
    return out

def select_top_publications(pubs: Optional[List[Any]], top: int) -> List[Dict[str, Any]]:
    if not pubs:
        return []
    scored = []
    for i, p in enumerate(pubs):
        link = _pub_get(p, "article_link", "") or ""
        has_link = 1 if (isinstance(link, str) and link.startswith(("http://", "https://"))) else 0
        scored.append((has_link, i, p))
    scored.sort(key=lambda t: (-t[0], t[1]))  # with-link first, then stable
    return [_pub_to_dict(p) for _, _, p in scored[:top]]


def format_profile_output(rec: Professor, top: int, only: str) -> Dict[str, Any]:
    email = getattr(rec, "پست_الکترونیکی", None)
    homepage = getattr(rec, "صفحه_شخصی", None)
    pubs = rec.publications if isinstance(rec.publications, list) else []

    out: Dict[str, Any] = {"name": rec.full_name()}
    if only in ("all", "contacts"):
        out["email"] = email
        out["homepage"] = homepage
    if only in ("all", "pubs"):
        out["top_publications"] = select_top_publications(pubs, top)
    return out

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Block 3 — Deterministic Persian profile lookup (Local)")
    ap.add_argument("--data", required=True, help="Path to normalized JSONL/JSON (output of Block 2)")
    ap.add_argument("--name", required=True, help="Professor full name in Persian (e.g., 'احمد آفتابی ثانی')")
    ap.add_argument("--top", type=int, default=5, help="Top-N publications to return")
    ap.add_argument("--only", choices=["all", "pubs", "contacts"], default="all", help="Limit output fields")
    ap.add_argument("--fuzzy", action="store_true", help="Allow minor Persian typos via RapidFuzz (optional)")
    args = ap.parse_args()

    # Load + index
    t0 = time.perf_counter()
    idx = ProfIndex()
    for rec in iter_jsonl_or_array(args.data):
        idx.add(rec)
    t1 = time.perf_counter()

    # Lookup
    rec = idx.find_exact(args.name)
    fuzzy_used = False
    if rec is None and args.fuzzy:
        matches = idx.find_fuzzy(args.name, limit=1, score_cutoff=92)
        if matches:
            _, _, rec = matches[0]
            fuzzy_used = True

    if rec is None:
        print(json.dumps({
            "ok": False,
            "reason": "not_found",
            "name_query": args.name
        }, ensure_ascii=False))
        return

    out = format_profile_output(rec, top=args.top, only=args.only)
    out_meta = {
        "ok": True,
        "lookup_ms": round((time.perf_counter() - t1) * 1000, 2),
        "load_ms": round((t1 - t0) * 1000, 2),
        "fuzzy": fuzzy_used
    }
    res = {"meta": out_meta, "result": out}
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
