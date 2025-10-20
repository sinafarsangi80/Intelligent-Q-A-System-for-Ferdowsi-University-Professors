"""
Block 8 — Hybrid retrieval (Local)
Dense (FAISS/LaBSE) + Lexical (BM25 or TF-IDF) fused via RRF.

Usage:
  # TF-IDF (default lexical),  top-5 fused
  python blocks/block8_hybrid.py \
    --artifacts artifacts \
    --corpus corpus/corpus.jsonl \
    --k 5 \
    --queries "تحلیل تیر تیموشنکو|الاستیسیته سایز-وابسته|گرادیان الاستیسیته"

  # Prefer BM25 (requires rank_bm25)
  python blocks/block8_hybrid.py \
    --artifacts artifacts \
    --corpus corpus/corpus.jsonl \
    --k 5 \
    --use_bm25 \
    --queries_file queries.txt
"""

import argparse, json, os, sys, time, datetime, re, math, hashlib
from typing import Any, Dict, Iterable, List, Tuple, Optional

# ---- Embeddings + FAISS (LangChain) ----
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ---- Optional BM25 (LangChain wrapper over rank_bm25) ----
try:
    from langchain_community.retrievers import BM25Retriever
    HAVE_BM25 = True
except Exception:
    HAVE_BM25 = False

# ---- Optional TF-IDF (scikit) ----
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as sk_normalize
    import numpy as np
    HAVE_TFIDF = True
except Exception:
    HAVE_TFIDF = False


# -----------------------------
# Persian-friendly normalization (light; no URL/email touching)
# -----------------------------
DIACRITICS_RE = re.compile("[" "\u064B-\u065F" "\u0670" "\u06D6-\u06ED" "]")
TATWEEL_RE = re.compile("\u0640")
NBSP_RE = re.compile("[\u00A0\u2007\u202F]")
ZWS_RE = re.compile("[\u200B\u200E\u200F]")
ZWNJ_RE = re.compile("\u200C+")
MULTI_SPACE_RE = re.compile(r"\s+")
ZWNJ = "\u200C"

def unify_arabic_persian_letters(s: str) -> str:
    if not isinstance(s, str):
        return s
    return s.replace("\u064A", "\u06CC").replace("\u0643", "\u06A9")  # ي→ی, ك→ک

def normalize_fa(s: str) -> str:
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

# -----------------------------
# IO helpers (read corpus built in Block 5)
# -----------------------------
def read_jsonl_or_array(path: str) -> Iterable[Dict[str, Any]]:
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
                if not line: continue
                r = json.loads(line)
                if isinstance(r, dict):
                    yield r

def ensure_dir(d: str):
    if d:
        os.makedirs(d, exist_ok=True)

# -----------------------------
# Dense retriever (FAISS + LaBSE)
# -----------------------------
def load_dense(artifacts_dir: str, device: str):
    embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/LaBSE",
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 64},
    )
    dim = len(embed.embed_query("سلام"))
    assert dim == 768, f"Embedding dimension mismatch: {dim}"
    vs = FAISS.load_local(artifacts_dir, embed, allow_dangerous_deserialization=True)
    return vs, embed

def dense_search(vs: FAISS, query: str, k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
    res = vs.similarity_search_with_score(query, k=k)
    out = []
    for doc, score in res:
        md = doc.metadata or {}
        doc_id = md.get("id") or md.get("doc_id") or ""
        out.append((doc_id, float(score), md))
    return out

# -----------------------------
# Lexical retriever (BM25 preferred; fallback TF-IDF)
# -----------------------------
def build_bm25(docs: List[Document], k: int = 50):
    # LangChain BM25 supports a preprocess_func
    def preproc(text: str) -> str:
        return normalize_fa(text)
    retriever = BM25Retriever.from_documents(docs, preprocess_func=preproc)
    retriever.k = k
    return retriever

class TFIDFLexical:
    def __init__(self, texts: List[str], metas: List[Dict[str,Any]], ngram=(1,2)):
        # Persian-aware token pattern (letters/digits underscore)
        token_pat = r"(?u)[\w\u0600-\u06FF]+"
        self.vec = TfidfVectorizer(
            analyzer="word",
            token_pattern=token_pat,
            ngram_range=ngram,
            min_df=1,
            lowercase=False,
            preprocessor=normalize_fa,
        )
        self.X = self.vec.fit_transform(texts)  # shape (N_docs, N_terms)
        # L2 normalize rows for cosine with fast dot
        self.X = sk_normalize(self.X, norm="l2", copy=False)
        self.metas = metas
        self.ids = [m.get("id") or m.get("doc_id") or f"idx_{i}" for i, m in enumerate(metas)]

    def search(self, query: str, k: int = 50) -> List[Tuple[str, float, Dict[str, Any]]]:
        qv = self.vec.transform([query])  # uses same preprocessor
        # Cosine = normalized => just X dot qv.T
        scores = (self.X @ qv.T).toarray().ravel()
        # top-k argsort
        import numpy as np
        if k >= len(scores):
            top_idx = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, k)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
        out = []
        for i in top_idx:
            out.append((self.ids[i], float(scores[i]), self.metas[i]))
        return out

# -----------------------------
# RRF fusion
# -----------------------------
def rrf_fuse(
    dense: List[Tuple[str, float, Dict[str,Any]]],
    lex:   List[Tuple[str, float, Dict[str,Any]]],
    k: int = 5,
    k_rrf: int = 60
) -> List[Tuple[str, float, Dict[str,Any], Dict[str,int]]]:
    """
    Inputs: lists of (doc_id, score, metadata) from two rankers.
    We ignore raw scores and fuse by ranks:
        RRF(doc) = sum(1 / (k_rrf + rank_i))
    Returns top-k as (doc_id, fused_score, metadata, ranks_dict).
    """
    # Build rank maps
    r_dense = {doc_id: r for r, (doc_id, _, _) in enumerate(dense, start=1)}
    r_lex   = {doc_id: r for r, (doc_id, _, _) in enumerate(lex,   start=1)}

    # All candidates
    all_ids = set(r_dense) | set(r_lex)
    fused = []
    for doc_id in all_ids:
        rd = r_dense.get(doc_id, 10**9)
        rl = r_lex.get(doc_id,   10**9)
        score = (1.0 / (k_rrf + rd)) + (1.0 / (k_rrf + rl))
        # Prefer metadata from the source that ranked it best
        if rd <= rl:
            md = next((m for i,(d,_,m) in enumerate(dense) if d==doc_id), {})
        else:
            md = next((m for i,(d,_,m) in enumerate(lex) if d==doc_id), {})
        fused.append((doc_id, float(score), md, {"dense_rank": rd if rd<10**9 else None,
                                                 "lex_rank":   rl if rl<10**9 else None}))
    # Sort by fused score desc
    fused.sort(key=lambda t: -t[1])
    return fused[:k]

# -----------------------------
# Utilities
# -----------------------------
def build_documents_for_lex(corpus_path: str) -> Tuple[List[Document], List[str], List[Dict[str,Any]]]:
    docs, texts, metas = [], [], []
    for d in read_jsonl_or_array(corpus_path):
        text = d.get("text") or " ".join(filter(None,[d.get("title"), d.get("venue")]))
        meta = {
            "id": d.get("id"),
            "doc_type": d.get("doc_type"),
            "title": d.get("title"),
            "venue": d.get("venue"),
            "article_link": (d.get("provenance") or {}).get("article_link"),
            "parent_id": d.get("parent_id"),
            "prof_id": d.get("prof_id"),
            "pub_id": d.get("pub_id"),
        }
        docs.append(Document(page_content=text, metadata=meta))
        texts.append(text)
        metas.append(meta)
    return docs, texts, metas

def to_safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Block 8 — Hybrid retrieval with RRF fusion (Local)")
    ap.add_argument("--artifacts", required=True, help="FAISS artifacts directory")
    ap.add_argument("--corpus", required=True, help="Path to Block 5 corpus (JSONL/JSON)")
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    ap.add_argument("--k", type=int, default=5, help="Top-k fused results to keep")
    ap.add_argument("--k_dense", type=int, default=50, help="How many candidates to pull from FAISS")
    ap.add_argument("--k_lex", type=int, default=100, help="How many candidates to pull from lexical")
    ap.add_argument("--k_rrf", type=int, default=60, help="RRF k (stabilizer)")
    ap.add_argument("--use_bm25", action="store_true", help="Use BM25 (needs rank_bm25). Otherwise TF-IDF.")
    ap.add_argument("--queries", default="", help="Queries separated by '|'")
    ap.add_argument("--queries_file", default="", help="UTF-8 file with one query per line")
    ap.add_argument("--reports_dir", default="reports", help="Output folder for JSON report")
    args = ap.parse_args()

    ensure_dir(args.reports_dir)

    # ---- Load dense (FAISS/LaBSE)
    t0 = time.perf_counter()
    vs, embed = load_dense(args.artifacts, args.device)
    t1 = time.perf_counter()

    # ---- Build lexical
    docs, texts, metas = build_documents_for_lex(args.corpus)
    if args.use_bm25 and HAVE_BM25:
        lex = build_bm25(docs, k=args.k_lex)
        lex_mode = "bm25"
    else:
        if not HAVE_TFIDF:
            raise RuntimeError("scikit-learn missing. Install: pip install scikit-learn")
        lex = TFIDFLexical(texts, metas, ngram=(1,2))
        lex_mode = "tfidf"

    # ---- Queries
    if args.queries.strip():
        queries = [q.strip() for q in args.queries.split("|") if q.strip()]
    elif args.queries_file:
        with open(args.queries_file, "r", encoding="utf-8") as f:
            queries = [ln.strip() for ln in f if ln.strip()]
    else:
        queries = [
            "تحلیل تیر تیموشنکو",
            "گرادیان الاستیسیته",
            "الاستیسیته سایز-وابسته",
            "روش های حل معادلات دیفرانسیل در تیر",
            "مهندسی عمران",
            "International Journal of Solids and Structures مقالات",
        ]

    # ---- Run hybrid retrieval
    report_queries = []
    for q in queries:
        # dense
        d_start = time.perf_counter()
        dense_res = dense_search(vs, q, k=args.k_dense)
        d_ms = round((time.perf_counter() - d_start)*1000, 2)

        # lexical
        l_start = time.perf_counter()
        if lex_mode == "bm25":
            # BM25 retriever returns list[Document]
            docs_lex = lex.get_relevant_documents(normalize_fa(q))
            lex_res = []
            for i, doc in enumerate(docs_lex[:args.k_lex], start=1):
                md = doc.metadata or {}
                lex_res.append((md.get("id") or "", float(1.0/(i+1)), md))  # pseudo-score (rank-based)
            l_ms = round((time.perf_counter() - l_start)*1000, 2)
        else:
            # TF-IDF
            lex_res = lex.search(q, k=args.k_lex)
            l_ms = round((time.perf_counter() - l_start)*1000, 2)

        # fused
        fused = rrf_fuse(dense_res, lex_res, k=args.k, k_rrf=args.k_rrf)

        # Build JSON-safe result rows
        def row_tuples(rows):
            out = []
            for (doc_id, score, md) in rows:
                out.append({
                    "id": doc_id or md.get("id"),
                    "score": to_safe_float(score),
                    "doc_type": md.get("doc_type"),
                    "title": md.get("title"),
                    "venue": md.get("venue"),
                    "article_link": md.get("article_link"),
                    "parent_id": md.get("parent_id"),
                    "prof_id": md.get("prof_id"),
                    "pub_id": md.get("pub_id"),
                })
            return out

        fused_rows = []
        for doc_id, fscore, md, ranks in fused:
            fused_rows.append({
                "id": doc_id or md.get("id"),
                "fused_score": to_safe_float(fscore),
                "dense_rank": ranks.get("dense_rank"),
                "lex_rank": ranks.get("lex_rank"),
                "doc_type": md.get("doc_type"),
                "title": md.get("title"),
                "venue": md.get("venue"),
                "article_link": md.get("article_link"),
                "parent_id": md.get("parent_id"),
                "prof_id": md.get("prof_id"),
                "pub_id": md.get("pub_id"),
            })

        report_queries.append({
            "query": q,
            "dense_ms": d_ms,
            "lex_ms": l_ms,
            "dense_top": row_tuples(dense_res[:args.k]),
            "lex_top":   row_tuples(lex_res[:args.k]),
            "fused_top": fused_rows
        })

    # ---- Report
    ntotal = getattr(vs.index, "ntotal", None)
    meta_path = os.path.join(args.artifacts, "index_meta.json")
    meta = {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        pass

    report = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "artifacts_dir": os.path.abspath(args.artifacts),
        "corpus": os.path.abspath(args.corpus),
        "index": {
            "ntotal": ntotal,
            "dim": 768,
            "load_ms": round((t1 - t0)*1000, 2)
        },
        "lexical_mode": lex_mode,
        "rrf_k": args.k_rrf,
        "k_fused": args.k,
        "k_dense": args.k_dense,
        "k_lex": args.k_lex,
        "meta": meta,
        "queries": report_queries
    }

    ensure_dir(args.reports_dir)
    fname = f"hybrid_{datetime.datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S')}.json"
    out_path = os.path.join(args.reports_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"✅ Hybrid report written to {out_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()



if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()


# -----------------------------
# RRF fusion
# -----------------------------
def rrf_fuse(
    dense: List[Tuple[str, float, Dict[str,Any]]],
    lex:   List[Tuple[str, float, Dict[str,Any]]],
    k: int = 5,
    k_rrf: int = 60
) -> List[Tuple[str, float, Dict[str,Any], Dict[str,int]]]:
    """
    Inputs: lists of (doc_id, score, metadata) from two rankers.
    We ignore raw scores and fuse by ranks:
        RRF(doc) = sum(1 / (k_rrf + rank_i))
    Returns top-k as (doc_id, fused_score, metadata, ranks_dict).
    """
    # Build rank maps
    r_dense = {doc_id: r for r, (doc_id, _, _) in enumerate(dense, start=1)}
    r_lex   = {doc_id: r for r, (doc_id, _, _) in enumerate(lex,   start=1)}

    # All candidates
    all_ids = set(r_dense) | set(r_lex)
    fused = []
    for doc_id in all_ids:
        rd = r_dense.get(doc_id, 10**9)
        rl = r_lex.get(doc_id,   10**9)
        score = (1.0 / (k_rrf + rd)) + (1.0 / (k_rrf + rl))
        # Prefer metadata from the source that ranked it best
        if rd <= rl:
            md = next((m for i,(d,_,m) in enumerate(dense) if d==doc_id), {})
        else:
            md = next((m for i,(d,_,m) in enumerate(lex) if d==doc_id), {})
        fused.append((doc_id, float(score), md, {"dense_rank": rd if rd<10**9 else None,
                                                 "lex_rank":   rl if rl<10**9 else None}))
    # Sort by fused score desc
    fused.sort(key=lambda t: -t[1])
    return fused[:k]

# -----------------------------
# Utilities
# -----------------------------
def build_documents_for_lex(corpus_path: str) -> Tuple[List[Document], List[str], List[Dict[str,Any]]]:
    docs, texts, metas = [], [], []
    for d in read_jsonl_or_array(corpus_path):
        text = d.get("text") or " ".join(filter(None,[d.get("title"), d.get("venue")]))
        meta = {
            "id": d.get("id"),
            "doc_type": d.get("doc_type"),
            "title": d.get("title"),
            "venue": d.get("venue"),
            "article_link": (d.get("provenance") or {}).get("article_link"),
            "parent_id": d.get("parent_id"),
            "prof_id": d.get("prof_id"),
            "pub_id": d.get("pub_id"),
        }
        docs.append(Document(page_content=text, metadata=meta))
        texts.append(text)
        metas.append(meta)
    return docs, texts, metas

def to_safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Block 8 — Hybrid retrieval with RRF fusion (Local)")
    ap.add_argument("--artifacts", required=True, help="FAISS artifacts directory")
    ap.add_argument("--corpus", required=True, help="Path to Block 5 corpus (JSONL/JSON)")
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    ap.add_argument("--k", type=int, default=5, help="Top-k fused results to keep")
    ap.add_argument("--k_dense", type=int, default=50, help="How many candidates to pull from FAISS")
    ap.add_argument("--k_lex", type=int, default=100, help="How many candidates to pull from lexical")
    ap.add_argument("--k_rrf", type=int, default=60, help="RRF k (stabilizer)")
    ap.add_argument("--use_bm25", action="store_true", help="Use BM25 (needs rank_bm25). Otherwise TF-IDF.")
    ap.add_argument("--queries", default="", help="Queries separated by '|'")
    ap.add_argument("--queries_file", default="", help="UTF-8 file with one query per line")
    ap.add_argument("--reports_dir", default="reports", help="Output folder for JSON report")
    args = ap.parse_args()

    ensure_dir(args.reports_dir)

    # ---- Load dense (FAISS/LaBSE)
    t0 = time.perf_counter()
    vs, embed = load_dense(args.artifacts, args.device)
    t1 = time.perf_counter()

    # ---- Build lexical
    docs, texts, metas = build_documents_for_lex(args.corpus)
    if args.use_bm25 and HAVE_BM25:
        lex = build_bm25(docs, k=args.k_lex)
        lex_mode = "bm25"
    else:
        if not HAVE_TFIDF:
            raise RuntimeError("scikit-learn missing. Install: pip install scikit-learn")
        lex = TFIDFLexical(texts, metas, ngram=(1,2))
        lex_mode = "tfidf"

    # ---- Queries
    if args.queries.strip():
        queries = [q.strip() for q in args.queries.split("|") if q.strip()]
    elif args.queries_file:
        with open(args.queries_file, "r", encoding="utf-8") as f:
            queries = [ln.strip() for ln in f if ln.strip()]
    else:
        queries = [
            "تحلیل تیر تیموشنکو",
            "گرادیان الاستیسیته",
            "الاستیسیته سایز-وابسته",
            "روش های حل معادلات دیفرانسیل در تیر",
            "مهندسی عمران",
            "International Journal of Solids and Structures مقالات",
        ]

    # ---- Run hybrid retrieval
    report_queries = []
    for q in queries:
        # dense
        d_start = time.perf_counter()
        dense_res = dense_search(vs, q, k=args.k_dense)
        d_ms = round((time.perf_counter() - d_start)*1000, 2)

        # lexical
        l_start = time.perf_counter()
        if lex_mode == "bm25":
            # BM25 retriever returns list[Document]
            docs_lex = lex.get_relevant_documents(normalize_fa(q))
            lex_res = []
            for i, doc in enumerate(docs_lex[:args.k_lex], start=1):
                md = doc.metadata or {}
                lex_res.append((md.get("id") or "", float(1.0/(i+1)), md))  # pseudo-score (rank-based)
            l_ms = round((time.perf_counter() - l_start)*1000, 2)
        else:
            # TF-IDF
            lex_res = lex.search(q, k=args.k_lex)
            l_ms = round((time.perf_counter() - l_start)*1000, 2)

        # fused
        fused = rrf_fuse(dense_res, lex_res, k=args.k, k_rrf=args.k_rrf)

        # Build JSON-safe result rows
        def row_tuples(rows):
            out = []
            for (doc_id, score, md) in rows:
                out.append({
                    "id": doc_id or md.get("id"),
                    "score": to_safe_float(score),
                    "doc_type": md.get("doc_type"),
                    "title": md.get("title"),
                    "venue": md.get("venue"),
                    "article_link": md.get("article_link"),
                    "parent_id": md.get("parent_id"),
                    "prof_id": md.get("prof_id"),
                    "pub_id": md.get("pub_id"),
                })
            return out

        fused_rows = []
        for doc_id, fscore, md, ranks in fused:
            fused_rows.append({
                "id": doc_id or md.get("id"),
                "fused_score": to_safe_float(fscore),
                "dense_rank": ranks.get("dense_rank"),
                "lex_rank": ranks.get("lex_rank"),
                "doc_type": md.get("doc_type"),
                "title": md.get("title"),
                "venue": md.get("venue"),
                "article_link": md.get("article_link"),
                "parent_id": md.get("parent_id"),
                "prof_id": md.get("prof_id"),
                "pub_id": md.get("pub_id"),
            })

        report_queries.append({
            "query": q,
            "dense_ms": d_ms,
            "lex_ms": l_ms,
            "dense_top": row_tuples(dense_res[:args.k]),
            "lex_top":   row_tuples(lex_res[:args.k]),
            "fused_top": fused_rows
        })

    # ---- Report
    ntotal = getattr(vs.index, "ntotal", None)
    meta_path = os.path.join(args.artifacts, "index_meta.json")
    meta = {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        pass

    report = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "artifacts_dir": os.path.abspath(args.artifacts),
        "corpus": os.path.abspath(args.corpus),
        "index": {
            "ntotal": ntotal,
            "dim": 768,
            "load_ms": round((t1 - t0)*1000, 2)
        },
        "lexical_mode": lex_mode,
        "rrf_k": args.k_rrf,
        "k_fused": args.k,
        "k_dense": args.k_dense,
        "k_lex": args.k_lex,
        "meta": meta,
        "queries": report_queries
    }

    ensure_dir(args.reports_dir)
    fname = f"hybrid_{datetime.datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S')}.json"
    out_path = os.path.join(args.reports_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"✅ Hybrid report written to {out_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
