import argparse, json, os, sys, re
from typing import Any, Dict, List, Union, Iterable

# Optional deps: persiantools (digits), hazm (not required here)
try:
    from persiantools.digits import fa_to_en
except Exception:
    fa_to_en = None  # We'll fall back to our own map if missing.

# ---------- Regexes for protection (preserve as-is) ----------
EMAIL_RE = re.compile(
    r"""(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b"""
)
URL_RE = re.compile(
    r"""(?ix)
    \b(
      (?:https?://|ftp://|file://)        # scheme
      [^\s<>"]+                            # the rest
      |                                    # or schemeless common form
      (?:www\.)[^\s<>"]+
    )\b
    """
)

# Placeholders (unlikely to appear in natural data)
PH_URL = "\u241fURL\u241f"
PH_MAIL = "\u241fMAIL\u241f"

# ---------- Character classes ----------
DIACRITICS_RE = re.compile(
    "["                         # Arabic diacritics + superscript alef
    "\u064B-\u065F"             # Fathatan..Wasla
    "\u0670"                    # Superscript Alef
    "\u06D6-\u06ED"             # Quranic marks range
    "]"
)

TATWEEL_RE = re.compile("\u0640")       # kashida
NBSP_RE = re.compile("[\u00A0\u2007\u202F]")  # various non-breaking spaces
ZWS_RE = re.compile("[\u200B\u200E\u200F]")   # ZWSP & directional marks (remove)
MULTI_SPACE_RE = re.compile(r"\s+")

# ZWNJ policy: keep existing, collapse repeats, remove spaces around it.
ZWNJ = "\u200C"
ZWNJ_RE = re.compile("\u200C+")

# Arabic → Persian letter unification
def unify_arabic_persian_letters(s: str) -> str:
    return (
        s.replace("\u064A", "\u06CC")  # ي → ی
         .replace("\u0643", "\u06A9")  # ك → ک
    )

# Persian/Arabic digits → ASCII 0-9
DIGIT_MAP = {ord(c): ord("0") + i for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}  # \u06F0-\u06F9
DIGIT_MAP.update({ord(c): ord("0") + i for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")})  # \u0660-\u0669

def digits_fa_to_en(s: str) -> str:
    if fa_to_en is not None:
        return fa_to_en(s)
    return s.translate(DIGIT_MAP)

def _protect(text: str) -> str:
    # Replace URLs/emails with placeholders and stash originals in order.
    protected: List[str] = []

    def _rep_url(m):
        protected.append(m.group(0))
        return PH_URL

    def _rep_mail(m):
        protected.append(m.group(0))
        return PH_MAIL

    # Important: preserve order—first URLs, then emails; we'll restore by scanning placeholders.
    text = URL_RE.sub(_rep_url, text)
    text = EMAIL_RE.sub(_rep_mail, text)
    return text, protected

def _unprotect(text: str, originals: List[str]) -> str:
    # Restore in the same order they were captured.
    out = []
    idx = 0
    i = 0
    while i < len(text):
        if text.startswith(PH_URL, i) or text.startswith(PH_MAIL, i):
            out.append(originals[idx])
            idx += 1
            i += len(PH_URL) if text.startswith(PH_URL, i) else len(PH_MAIL)
        else:
            out.append(text[i])
            i += 1
    return "".join(out)

def normalize_persian_text(s: str) -> str:
    # Short-circuit if entire string is a URL/email
    if URL_RE.fullmatch(s) or EMAIL_RE.fullmatch(s):
        return s

    # Protect embedded URLs/emails
    s_prot, originals = _protect(s)

    # Normalize
    s_norm = unify_arabic_persian_letters(s_prot)
    s_norm = DIACRITICS_RE.sub("", s_norm)
    s_norm = TATWEEL_RE.sub("", s_norm)

    # ZWNJ sanitization: remove ZWSP & dir marks, collapse ZWNJ runs, trim spaces around ZWNJ
    s_norm = ZWS_RE.sub("", s_norm)
    s_norm = ZWNJ_RE.sub(ZWNJ, s_norm)
    # remove spaces around ZWNJ
    s_norm = re.sub(r"\s*"+ZWNJ+r"\s*", ZWNJ, s_norm)

    # Whitespace: normalize NBSPs → space, collapse whitespace, trim
    s_norm = NBSP_RE.sub(" ", s_norm)
    s_norm = MULTI_SPACE_RE.sub(" ", s_norm).strip()

    # Digits: Persian/Arabic-Indic → ASCII
    s_norm = digits_fa_to_en(s_norm)

    # Restore protected pieces exactly
    s_final = _unprotect(s_norm, originals)
    return s_final

def normalize_any(x: Any) -> Any:
    if isinstance(x, str):
        return normalize_persian_text(x)
    if isinstance(x, list):
        return [normalize_any(v) for v in x]
    if isinstance(x, dict):
        # If a field is explicitly a URL/email string, it will be preserved by normalize_persian_text
        return {k: normalize_any(v) for k, v in x.items()}
    return x  # numbers, bools, None

def iter_input_records(path: str) -> Iterable[Dict[str, Any]]:
    # Accept JSONL (one obj per line) or JSON array file.
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        if head.lstrip().startswith("["):
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Top-level JSON must be a list of records.")
            for rec in data:
                if isinstance(rec, dict):
                    yield rec
        else:
            # JSONL
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if isinstance(rec, dict):
                    yield rec

def write_jsonl(records: Iterable[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        for rec in records:
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

def spot_checks(src_path: str, samples: int = 6) -> None:
    print("\n— Spot checks (before → after) —")
    c = 0
    for rec in iter_input_records(src_path):
        if c >= samples:
            break
        # Pick a few likely string fields if present
        candidates = []
        for key in ["نام", "نام خانوادگی", "دانشکده", "گروه آموزشی", "وضعیت اشتغال", "پست الکترونیکی", "صفحه شخصی", "title", "journal", "article_link"]:
            if key in rec and isinstance(rec[key], str):
                candidates.append((key, rec[key]))
        if not candidates:
            # fallback: first str in dict
            for k, v in rec.items():
                if isinstance(v, str):
                    candidates.append((k, v))
                    break

        print(f"\nRecord #{c+1}")
        for k, v in candidates[:3]:
            aft = normalize_persian_text(v)
            print(f"  {k}:")
            print(f"    before: {v}")
            print(f"    after : {aft}")
        c += 1
    if c == 0:
        print("No records found for spot check.")

def main():
    ap = argparse.ArgumentParser(description="Block 2 — Persian normalization (Local)")
    ap.add_argument("--in", dest="inp", required=True, help="Input JSON or JSONL path")
    ap.add_argument("--out", dest="out", default="data/normalized/records.jsonl", help="Output JSONL path")
    ap.add_argument("--samples", type=int, default=6, help="How many spot-check rows to print")
    args = ap.parse_args()

    # Spot checks from raw (before normalization)
    spot_checks(args.inp, samples=args.samples)

    # Normalize streamingly and write JSONL
    def _normalized_stream():
        for rec in iter_input_records(args.inp):
            yield normalize_any(rec)

    write_jsonl(_normalized_stream(), args.out)

    print(f"\n✅ Done. Wrote normalized JSONL → {args.out}")
    print("Verify that emails/URLs remained untouched in the spot checks above.")
    print("You can re-run with a larger --samples if needed.")

if __name__ == "__main__":
    main()
