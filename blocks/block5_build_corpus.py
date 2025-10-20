# -*- coding: utf-8 -*-
"""
Block 5 — Document assembly & chunking (Local) — tailored to professors_norm.json

Inputs:
  - JSONL or JSON array produced by Block 2 (normalized). Records include:
      "نام", "نام خانوادگی", "دانشکده", "گروه آموزشی", "پست الکترونیکی", "صفحه شخصی",
      "publications": [
        { "title", "authors": [...], "journal", "article_link" }
      ]

Outputs:
  - corpus/corpus.jsonl   (publication docs + optional chunk docs)
  - corpus/stats.json     (counts & sanity metrics)

Usage:
  python blocks/block5_build_corpus.py \
    --data data/normalized/records.jsonl \
    --out corpus/corpus.jsonl \
    --stats corpus/stats.json
"""

import argparse, json, os, re, hashlib, math, statistics
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -------------------------
# Minimal normalization (names only; DO NOT touch URLs/emails)
# -------------------------
DIACRITICS_RE = re.compile("[" "\u064B-\u065F" "\u0670" "\u06D6-\u06ED" "]")
TATWEEL_RE = re.compile("\u0640")
NBSP_RE = re.compile("[\u00A0\u2007\u202F]")
ZWS_RE = re.compile("[\u200B\u200E\u200F]")
ZWNJ_RE = re.compile("\u200C+")
MULTI_SPACE_RE = re.compile(r"\s+")
ZWNJ = "\u200C"

def unify_arabic_persian_letters(s: str) -> str:
    if not isinstance(s, str): return s
    return s.replace("\u064A", "\u06CC").replace("\u0643", "\u06A9")  # ي→ی, ك→ک

def normalize_name(s: str) -> str:
    if not isinstance(s, str): return ""
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

def ensure_dir_for(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

# -------------------------
# Stable IDs
# -------------------------
def sha1_12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def make_prof_id(first: str, last: str) -> str:
    full_norm = (normalize_name(first) + " " + normalize_name(last)).strip()
    return sha1_12(full_norm or "unknown")

def make_pub_id(title: str, journal: str, link: str) -> str:
    key = " | ".join([
        (title or "").strip(),
        (journal or "").strip(),
        (link or "").strip()
    ])
    return sha1_12(key or "unknown")

# -------------------------
# Chunking (character-based, Persian-friendly)
# -------------------------
DEFAULT_CHUNK_SIZE = 450     # middle of the 300–600 char window
DEFAULT_CHUNK_OVERLAP = 70
SEPARATORS = ["\n\n", "\n", "۔", ".", "؛", "،", " ", ""]  # Persian + generic

def need_chunk(text: str, chunk_size: int, overlap: int) -> bool:
    if not isinstance(text, str): return False
    return len(text) > (chunk_size + overlap)

def recursive_split(text: str, chunk_size: int, overlap: int) -> List[Tuple[int, int, str]]:
    """
    Returns list of tuples (start_offset, end_offset, chunk_text).
    A simple recursive splitter using SEPARATORS, biased to keep 300–600 chars.
    """
    if not text:
        return []
    spans: List[Tuple[int, int, str]] = []

    def _split_segment(seg_text: str, base_off: int):
        if len(seg_text) <= chunk_size + overlap:
            spans.append((base_off, base_off + len(seg_text), seg_text))
            return
        # Try separators to find a nice boundary near chunk_size
        cut = None
        target = chunk_size
        window = min(len(seg_text), chunk_size + 100)  # small slack
        for sep in SEPARATORS:
            if sep == "":
                continue
            idx = seg_text.rfind(sep, 0, window)
            if idx != -1 and idx >= int(chunk_size * 0.6):
                cut = idx + len(sep)
                break
        if cut is None:
            cut = min(len(seg_text), chunk_size)

        chunk = seg_text[:cut]
        spans.append((base_off, base_off + len(chunk), chunk))

        # Overlap
        start_next = max(0, len(chunk) - overlap)
        rest = seg_text[start_next:]
        _split_segment(rest, base_off + start_next)

    _split_segment(text, 0)
    # De-duplicate possible zero-length overlaps (edge cases)
    clean = []
    for s, e, t in spans:
        if e > s and t:
            clean.append((s, e, t))
    return clean

# -------------------------
# Build corpus
# -------------------------
def build_corpus(input_path: str,
                 out_path: str,
                 stats_path: str,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:

    ensure_dir_for(out_path)
    ensure_dir_for(stats_path)

    docs_written = 0
    pub_docs = 0
    chunk_docs = 0
    n_profs = 0
    n_pubs = 0

    lengths_title: List[int] = []
    lengths_journal: List[int] = []
    chunk_lengths: List[int] = []

    with open(out_path, "w", encoding="utf-8") as w:
        for rec_idx, prof in enumerate(iter_jsonl_or_array(input_path)):
            first = (prof.get("نام") or "").strip()
            last  = (prof.get("نام خانوادگی") or prof.get("نام_خانوادگی") or "").strip()
            faculty = (prof.get("دانشکده") or "").strip()
            group   = (prof.get("گروه آموزشی") or prof.get("گروه_آموزشی") or "").strip()
            email   = (prof.get("پست الکترونیکی") or "").strip()
            homepage= (prof.get("صفحه شخصی") or "").strip()

            n_profs += 1
            prof_full = f"{first} {last}".strip()
            prof_id = make_prof_id(first, last)

            pubs = prof.get("publications") or []
            if not isinstance(pubs, list):
                continue

            for p_idx, p in enumerate(pubs):
                # Accept dict-like & pydantic
                if hasattr(p, "model_dump"):   # pydantic v2
                    p = p.model_dump(exclude_none=True)
                elif hasattr(p, "dict"):       # pydantic v1
                    p = p.dict(exclude_none=True)
                if not isinstance(p, dict):
                    continue

                title = (p.get("title") or "").strip()
                authors_list = p.get("authors") if isinstance(p.get("authors"), list) else []
                authors_text = "، ".join([str(a) for a in authors_list if a])  # Persian comma
                journal = (p.get("journal") or "").strip()
                link = (p.get("article_link") or "").strip()

                pub_id = make_pub_id(title, journal, link)
                n_pubs += 1

                # Add basic length stats
                lengths_title.append(len(title))
                lengths_journal.append(len(journal))

                # ==== Publication-level doc (always) ====
                pub_doc_id = f"prof_{prof_id}__pub_{pub_id}"
                full_text = "\n".join([
                    title if title else "",
                    journal if journal else "",
                    ("نویسندگان: " + authors_text) if authors_text else "",
                    ("استاد: " + prof_full) if prof_full else "",
                    ("دانشکده: " + faculty) if faculty else "",
                    ("گروه آموزشی: " + group) if group else "",
                ]).strip()

                pub_doc = {
                    "doc_type": "publication",
                    "id": pub_doc_id,
                    "lang": "fa",  # corpus intended for FA/Persian queries
                    "prof_id": prof_id,
                    "pub_id": pub_id,
                    "provenance": {
                        "prof_full_name": prof_full,
                        "article_link": link or None,
                        "source_record_index": rec_idx,
                    },
                    "title": title or None,
                    "venue": journal or None,
                    "authors": authors_list or None,
                    "text": full_text or None,
                    "contacts": {
                        "email": email or None,
                        "homepage": homepage or None
                    },
                    "faculty": faculty or None,
                    "group": group or None,
                }
                w.write(json.dumps(pub_doc, ensure_ascii=False) + "\n")
                docs_written += 1
                pub_docs += 1

                # ==== Chunking policy (no abstract in your data) ====
                # Only chunk long *fields* (typically rare here). We consider:
                # - title
                # - journal
                # If either is very long (> chunk_size+overlap), cut into 300–600-char chunks.
                for field_name, field_value in (("title", title), ("journal", journal)):
                    if not field_value:
                        continue
                    if need_chunk(field_value, chunk_size, overlap):
                        spans = recursive_split(field_value, chunk_size, overlap)
                        for ci, (s, e, txt) in enumerate(spans, start=1):
                            chunk_id = f"{pub_doc_id}__{field_name}__c{ci:04d}"
                            chunk_doc = {
                                "doc_type": "chunk",
                                "id": chunk_id,
                                "lang": "fa",
                                "prof_id": prof_id,
                                "pub_id": pub_id,
                                "parent_id": pub_doc_id,
                                "source_field": field_name,
                                "offset_start": s,
                                "offset_end": e,
                                "text": txt,
                                "provenance": {
                                    "prof_full_name": prof_full,
                                    "title": title or None,
                                    "venue": journal or None
                                }
                            }
                            w.write(json.dumps(chunk_doc, ensure_ascii=False) + "\n")
                            docs_written += 1
                            chunk_docs += 1
                            chunk_lengths.append(len(txt))

    # -------------------------
    # Stats
    # -------------------------
    def _basic(vs: List[int]) -> Dict[str, Any]:
        if not vs:
            return {"count": 0, "avg": 0, "p50": 0, "p95": 0, "max": 0}
        vs_sorted = sorted(vs)
        def pct(p):  # p in [0,100]
            k = (len(vs_sorted)-1) * (p/100.0)
            f = math.floor(k); c = math.ceil(k)
            if f == c:
                return vs_sorted[int(k)]
            return vs_sorted[f] + (vs_sorted[c]-vs_sorted[f])*(k-f)
        return {
            "count": len(vs_sorted),
            "avg": round(statistics.mean(vs_sorted), 1),
            "p50": round(pct(50), 1),
            "p95": round(pct(95), 1),
            "max": max(vs_sorted)
        }

    stats = {
        "professors": n_profs,
        "publications": n_pubs,
        "docs_publication": pub_docs,
        "docs_chunk": chunk_docs,
        "lengths": {
            "title_chars": _basic(lengths_title),
            "journal_chars": _basic(lengths_journal),
            "chunk_chars": _basic(chunk_lengths),
        }
    }

    ensure_dir_for(stats_path)
    with open(stats_path, "w", encoding="utf-8") as s:
        json.dump(stats, s, ensure_ascii=False, indent=2)

    print(f"✅ Wrote corpus: {out_path} ({docs_written} docs; {pub_docs} publication docs, {chunk_docs} chunk docs)")
    print(f"📊 Wrote stats : {stats_path}")
    if chunk_docs == 0:
        print("ℹ️ No chunks were created (fields were not long enough). This is OK for your schema.")


def main():
    ap = argparse.ArgumentParser(description="Block 5 — Build RAG corpus from professors_norm.json")
    ap.add_argument("--data", required=True, help="Path to normalized JSONL/JSON (Block 2 output)")
    ap.add_argument("--out", default="corpus/corpus.jsonl", help="Output JSONL path")
    ap.add_argument("--stats", default="corpus/stats.json", help="Stats JSON path")
    ap.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size in chars")
    ap.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap in chars")
    args = ap.parse_args()

    build_corpus(args.data, args.out, args.stats, args.chunk_size, args.overlap)

if __name__ == "__main__":
    main()
