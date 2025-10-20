# -*- coding: utf-8 -*-
"""
Block 7 — Load FAISS & smoke tests (Local) — JSON reports

Usage:
  python blocks/block7_smoke_test.py \
    --artifacts artifacts \
    --k 5 \
    --queries "تحلیل تیر تیموشنکو|گرادیان الاستیسیته در مکانیک جامدات"
"""

import argparse, json, os, sys, time, datetime

# ---- Embeddings loader ----
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


def load_meta(meta_path: str) -> dict:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def main():
    ap = argparse.ArgumentParser(description="Block 7 — FAISS smoke tests (report mode)")
    ap.add_argument("--artifacts", default="artifacts", help="Dir with index.faiss, index.pkl, index_meta.json")
    ap.add_argument("--k", type=int, default=5, help="Top-k neighbors")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Embedding device")
    ap.add_argument("--queries", default="", help="Queries separated by '|'")
    ap.add_argument("--queries_file", default="", help="File with queries (one per line, UTF-8)")
    ap.add_argument("--reports_dir", default="reports", help="Output reports folder")
    args = ap.parse_args()

    os.makedirs(args.reports_dir, exist_ok=True)

    # ---- Load meta ----
    meta_path = os.path.join(args.artifacts, "index_meta.json")
    meta = load_meta(meta_path)

    # ---- Load embeddings ----
    embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/LaBSE",
        model_kwargs={"device": args.device},
        encode_kwargs={"batch_size": 64},
    )
    dim = len(embed.embed_query("سلام"))
    assert dim == 768, f"Embedding dimension mismatch: {dim}"

    # ---- Load FAISS ----
    t0 = time.perf_counter()
    vs = FAISS.load_local(args.artifacts, embed, allow_dangerous_deserialization=True)
    t1 = time.perf_counter()
    ntotal = getattr(vs.index, "ntotal", None)

    # ---- Collect queries ----
    queries = []
    if args.queries.strip():
        queries = [q.strip() for q in args.queries.split("|") if q.strip()]
    elif args.queries_file:
        with open(args.queries_file, "r", encoding="utf-8") as f:
            queries = [ln.strip() for ln in f if ln.strip()]
    if not queries:
        queries = [
            "تحلیل تیر تیموشنکو",
            "گرادیان الاستیسیته",
            "مهندسی عمران",
            "Dynamic analysis of beams",
            "مقالات مکانیک جامدات",
        ]

    # ---- Run searches ----
    results = []
    for q in queries:
        tqs = time.perf_counter()
        hits = vs.similarity_search_with_score(q, k=args.k)
        tqe = time.perf_counter()
        res_q = {
            "query": q,
            "elapsed_ms": round((tqe - tqs) * 1000, 2),
            "results": []
        }
        for doc, score in hits:
            md = doc.metadata or {}
            res_q["results"].append({
                "score": float(score),   # <-- cast to Python float
                "doc_type": md.get("doc_type"),
                "title": md.get("title"),
                "venue": md.get("venue"),
                "article_link": md.get("article_link") or md.get("provenance.article_link"),
                "parent_id": md.get("parent_id"),
                "id": md.get("id"),
            })
        results.append(res_q)

    # ---- Assemble report ----
    report = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "artifacts": os.path.abspath(args.artifacts),
        "meta": meta,
        "index": {
            "ntotal": ntotal,
            "dim": dim,
            "load_ms": round((t1 - t0) * 1000, 2),
        },
        "queries": results,
    }

    # ---- Write JSON ----
    fname = f"smoke_{datetime.datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S')}.json"
    out_path = os.path.join(args.reports_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"✅ Report written to {out_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
