# -*- coding: utf-8 -*-
import os, json
from typing import List, Optional, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# --- Import your blocks (we'll call their functions/classes) ---
# Block 3: deterministic lookup (Exact Persian match)  [A]
import blocks.block3_lookup as A  # ProfIndex, format_profile_output, select_top_publications
# Block 4: intent detection helpers (regex rules)      [Router]
import blocks.block4_router as R  # normalize_text, extract_track_a
# Block 8: hybrid retrieval (dense + lexical + RRF)    [B]
import blocks.block8_hybrid as B  # load_dense, dense_search, TFIDFLexical, build_bm25, rrf_fuse, build_documents_for_lex

# --- LLM (DeepSeek-R1 via Ollama) ---
from api.llm_clients import r1_generate, r1_generate_stream

# ---------- Config ----------
DATA_PATH     = os.getenv("RAG_DATA", "data/normalized/records.jsonl")   # output of Block 2
CORPUS_PATH   = os.getenv("RAG_CORPUS", "corpus/corpus.jsonl")           # output of Block 5
ARTIFACTS_DIR = os.getenv("RAG_ARTIFACTS", "artifacts")                  # output of Block 6
DEVICE        = os.getenv("RAG_DEVICE", "cpu")                           # "cpu" or "cuda"

# ---------- App ----------
app = FastAPI(title="Sina RAG — Block 12 API", version="0.4.0")

# ---------- Globals (loaded on startup) ----------
A_INDEX: Optional[A.ProfIndex] = None
DENSE_VS = None
LEX_OBJ  = None
LEX_MODE = "bm25"  # or "tfidf"

# NEW: lightweight name resolver store
PROF_NORM_TO_DISPLAY = {}   # normalized -> canonical display
PROF_DISPLAY_TO_NORM = {}   # display -> normalized

def _extract_display_name(rec: dict) -> Optional[str]:
    # Try normalized pipeline name if present; otherwise compose from Persian fields.
    n = rec.get("name")
    if n:
        return n.strip()
    first = (rec.get("نام") or "").strip()
    last  = (rec.get("نام خانوادگی") or "").strip()
    combo = f"{first} {last}".strip()
    return combo if combo else None

def _norm(s: str) -> str:
    return R.normalize_text(s or "")

# ---------- Pydantic IO ----------
class DeterministicRequest(BaseModel):
    name: str = Field(..., description="Full professor name in Persian, e.g. 'احمد آفتابی ثانی'")
    top: int = 5
    only: str = Field("all", pattern="^(all|pubs|contacts)$")

class DeterministicResponse(BaseModel):
    ok: bool
    result: Optional[Dict[str, Any]] = None
    detail: Optional[str] = None

class RagRequest(BaseModel):
    query: str
    k: int = 5
    k_dense: int = 50
    k_lex: int = 100
    k_rrf: int = 60
    fused_threshold: Optional[float] = 0.03

class Hit(BaseModel):
    id: Optional[str]
    fused_score: Optional[float] = None
    score: Optional[float] = None
    doc_type: Optional[str] = None
    title: Optional[str] = None
    venue: Optional[str] = None
    article_link: Optional[str] = None
    parent_id: Optional[str] = None
    prof_id: Optional[str] = None
    pub_id: Optional[str] = None
    dense_rank: Optional[int] = None
    lex_rank: Optional[int] = None
    text: Optional[str] = None  

class RagResponse(BaseModel):
    ok: bool
    query: str
    fused_top: List[Hit] = []
    dense_top: List[Hit] = []
    lex_top: List[Hit] = []
    meta: Dict[str, Any] = {}

class ChatRequest(BaseModel):
    query: str
    # optional knobs reused by RAG
    k: int = 5
    k_dense: int = 50
    k_lex: int = 100
    k_rrf: int = 60
    top_pubs: int = 5
    fused_threshold: Optional[float] = 0.03

class ChatResponse(BaseModel):
    route: str                       # "A", "B", or "clarify"
    answer: Optional[str] = None     # short Persian-friendly text to render in UI
    data: Optional[Dict[str, Any]] = None
    sources: List[Hit] = []

# ---------- Helpers ----------
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _row(doc_id, score, md):
    return Hit(
        id=doc_id or md.get("id"),
        score=_safe_float(score),
        doc_type=md.get("doc_type"),
        title=md.get("title"),
        venue=md.get("venue"),
        article_link=md.get("article_link"),
        parent_id=md.get("parent_id"),
        prof_id=md.get("prof_id"),
        pub_id=md.get("pub_id"),
        text=md.get("text") or md.get("chunk") or md.get("content")  
    )

def _fused_row(doc_id, fscore, md, ranks):
    return Hit(
        id=doc_id or md.get("id"),
        fused_score=_safe_float(fscore),
        dense_rank=ranks.get("dense_rank"),
        lex_rank=ranks.get("lex_rank"),
        doc_type=md.get("doc_type"),
        title=md.get("title"),
        venue=md.get("venue"),
        article_link=md.get("article_link"),
        parent_id=md.get("parent_id"),
        prof_id=md.get("prof_id"),
        pub_id=md.get("pub_id"),
        text=md.get("text") or md.get("chunk") or md.get("content")  
    )

# ---------- SSE helpers ----------
def _sse_pack(obj: Dict[str, Any]) -> str:
    # ensure_ascii=False to keep Persian chars intact
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

def _stream_deterministic_answer(ans_text: str):
    """Yield answer lines as streaming deltas."""
    for line in ans_text.splitlines(True):
        yield _sse_pack({"type": "token", "delta": line})
    yield _sse_pack({"type": "done"})

# ---- LLM grounding helpers (Track B) ----
SYSTEM_MSG = (
    "تو یک دستیار پرسش‌وپاسخ «استخراجی» هستی. فقط و فقط از متن «منابع» استفاده کن. "
    "هر جملهٔ پاسخ باید با ارجاع [#] ختم شود. اگر پاسخ دقیق در منابع نیست، بگو «اطلاعات کافی نیست». "
    "هیچ دانش بیرونی اضافه نکن و لینک/عنوان را تحریف نکن. "
    "مهم: پاسخ را «فقط به فارسی» بنویس؛ از واژگان انگلیسی استفاده نکن مگر برای نام ژورنال‌ها، نام‌های proper و لینک‌ها."
)

def _shorten(s: str, limit: int = 400) -> str:
    if not s: return ""
    s = s.replace("\n", " ").strip()
    return (s[:limit] + "…") if len(s) > limit else s

def _ctx_from_hits(hits: List[Hit]) -> str:
    """
    Build grounded context with brief snippets. Keep it small for 1.5B models.
    Format:
      [#] title — venue — link
          snippet
    """
    lines = []
    for i, h in enumerate(hits, start=1):
        head_parts = []
        if h.title: head_parts.append(h.title)
        if h.venue: head_parts.append(h.venue)
        if h.article_link: head_parts.append(h.article_link)
        head = " — ".join(head_parts) if head_parts else (h.id or f"doc-{i}")
        snippet = _shorten(h.text or "")
        if snippet:
            lines.append(f"[{i}] {head}\n    {snippet}")
        else:
            lines.append(f"[{i}] {head}")
    return "\n".join(lines)

# ---------- Startup ----------
@app.on_event("startup")
def _startup():
    global A_INDEX, DENSE_VS, LEX_OBJ, LEX_MODE, PROF_NORM_TO_DISPLAY, PROF_DISPLAY_TO_NORM

    # Track A: load deterministic index (exact Persian)  — ProfIndex from Block 3
    idx = A.ProfIndex()
    PROF_NORM_TO_DISPLAY = {}
    PROF_DISPLAY_TO_NORM = {}

    for rec in A.iter_jsonl_or_array(DATA_PATH):
        idx.add(rec)
        # capture display + normalized variants for fuzzy/partial resolution
        disp = _extract_display_name(rec)
        if disp:
            nd = _norm(disp)
            PROF_NORM_TO_DISPLAY[nd] = disp
            PROF_DISPLAY_TO_NORM[disp] = nd

    A_INDEX = idx

    # Track B loaders...
    vs, _ = B.load_dense(ARTIFACTS_DIR, DEVICE)
    DENSE_VS = vs

    docs, texts, metas = B.build_documents_for_lex(CORPUS_PATH)
    try:
        LEX_OBJ = B.TFIDFLexical(texts, metas, ngram=(1, 2))
        LEX_MODE = "tfidf"
    except Exception:
        if hasattr(B, "BM25Retriever") and B.HAVE_BM25:
            LEX_OBJ = B.build_bm25(docs, k=100)
            LEX_MODE = "bm25"
        else:
            raise RuntimeError("No lexical retriever available.")
        
def _resolve_prof_name_flex(input_name: str) -> Optional[str]:
    """
    Tries to resolve a partial or surname-only input to a single canonical display name.
    Returns display name if unique; otherwise None.
    """
    if not input_name:
        return None

    qn = _norm(input_name)

    # 1) Exact (normalized) match
    if qn in PROF_NORM_TO_DISPLAY:
        return PROF_NORM_TO_DISPLAY[qn]

    # 2) Token / substring match across normalized names
    #    - prefer token match (word-level) first
    tokens = qn.split()
    cands = set()

    for n_norm, disp in PROF_NORM_TO_DISPLAY.items():
        # token-level match (e.g., 'صدوقی' equals a token in the full name)
        n_tokens = n_norm.split()
        if any(t and t in n_tokens for t in tokens if t):
            cands.add(disp)
            continue
        # fallback: substring (handles ZWNJ variants, etc.)
        if qn and qn in n_norm:
            cands.add(disp)

    if len(cands) == 1:
        return next(iter(cands))

    # 3) Not unique or not found
    return None

# ---------- Endpoint A: Deterministic ----------
@app.post("/deterministic", response_model=DeterministicResponse)
def deterministic(req: DeterministicRequest):
    rec = A_INDEX.find_exact(req.name) if A_INDEX else None
    if rec is None:
        return DeterministicResponse(ok=False, detail="not_found")
    out = A.format_profile_output(rec, top=req.top, only=req.only)
    return DeterministicResponse(ok=True, result=out)

# ---------- Endpoint B: RAG / hybrid retrieval ----------
@app.post("/rag", response_model=RagResponse)
def rag(req: RagRequest):
    # Dense (FAISS)
    dense_res = B.dense_search(DENSE_VS, req.query, k=req.k_dense)
    dense_top = [_row(did, sc, md) for (did, sc, md) in dense_res[:req.k]]

    # Lexical
    if LEX_MODE == "bm25":
        docs_lex = LEX_OBJ.get_relevant_documents(B.normalize_fa(req.query))
        lex_res = []
        for i, doc in enumerate(docs_lex[:req.k_lex], start=1):
            md = doc.metadata or {}
            lex_res.append((md.get("id") or "", float(1.0/(i+1)), md))
    else:
        lex_res = LEX_OBJ.search(req.query, k=req.k_lex)
    lex_top = [_row(did, sc, md) for (did, sc, md) in lex_res[:req.k]]

    # RRF fuse across *all* candidates so threshold can apply
    fuse_k = (len(dense_res) + len(lex_res)) if req.fused_threshold is not None else req.k
    fused_all = B.rrf_fuse(dense_res, lex_res, k=fuse_k, k_rrf=req.k_rrf)
    fused_rows = [_fused_row(did, fs, md, ranks) for (did, fs, md, ranks) in fused_all]

    if req.fused_threshold is not None:
        fused_top = [h for h in fused_rows
                     if (h.fused_score is not None and h.fused_score >= req.fused_threshold)]
    else:
        fused_top = fused_rows[:req.k]

    return RagResponse(
        ok=True,
        query=req.query,
        fused_top=fused_top,
        dense_top=dense_top,
        lex_top=lex_top,
        meta={
            "lexical_mode": LEX_MODE,
            "k_rrf": req.k_rrf,
            "fused_threshold": req.fused_threshold,
            "fused_returned": len(fused_top),
        },
    )


# ---------- Endpoint Chat: Router → A or B (with DeepSeek-R1 on B) ----------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Use Block 4’s intent extraction (Persian rules)
    intent, name = R.extract_track_a(req.query)  # ("email"|"homepage"|"pubs"|None, name|None)

    # If A-intent:
    if intent is not None:
        # 1) If no name captured, still clarify as before
        if not name:
            msg = "نام کامل استاد را دقیق وارد کنید (نام + نام خانوادگی) تا نتیجهٔ «{}» را ارائه کنم.".format(
                "ایمیل" if intent=="email" else ("صفحهٔ شخصی" if intent=="homepage" else "لیست مقالات")
            )
            return ChatResponse(route="clarify", answer=msg, data={"intent": intent}, sources=[])

        # 2) Try exact match first
        rec = A_INDEX.find_exact(name) if A_INDEX else None

        # 3) If exact not found, try flexible resolver (surname/partial)
        if rec is None:
            resolved = _resolve_prof_name_flex(name)
            if resolved:
                rec = A_INDEX.find_exact(resolved)

        # 4) If still none, but multiple possible candidates exist, list a few for clarification
        if rec is None:
            # Build candidate list using the same resolver logic but keeping all matches
            qn = _norm(name)
            candidates = []
            for n_norm, disp in PROF_NORM_TO_DISPLAY.items():
                n_tokens = n_norm.split()
                if qn in n_norm or qn in n_tokens:
                    candidates.append(disp)

            if candidates:
                # de-dup + cap to 7 options for UI
                candidates = sorted(set(candidates))[:7]
                return ChatResponse(
                    route="clarify",
                    answer="چند استاد با این نام پیدا شد؛ یکی را انتخاب کنید:",
                    data={"intent": intent, "candidates": candidates},
                    sources=[]
                )

            # Fallback: generic clarify
            return ChatResponse(
                route="clarify",
                answer="هیچ نام دقیقی پیدا نشد؛ لطفاً نام کامل فارسی استاد را بدهید.",
                data={"intent": intent},
                sources=[]
            )

        # 5) We have a record → proceed with your existing A flow
        result = A.format_profile_output(rec, top=req.top_pubs, only="all")
        ans = []
        if intent == "email" and result.get("email"):
            ans.append(f"ایمیل {result['name']}: {result['email']}")
        elif intent == "homepage" and result.get("homepage"):
            ans.append(f"صفحهٔ شخصی {result['name']}: {result['homepage']}")
        else:
            pubs = result.get("top_publications") or []
            ans.append(f"۵ مورد از مقالات {result['name']}:")
            for i, p in enumerate(pubs[:req.top_pubs], start=1):
                t = p.get("title") or ""
                v = p.get("journal") or ""
                link = p.get("article_link") or ""
                ans.append(f"{i}. {t} — {v} {'('+link+')' if link else ''}")

        srcs: List[Hit] = []
        if result.get("homepage"):
            srcs.append(Hit(id="homepage", title="صفحه شخصی", venue=result["name"], article_link=result["homepage"]))
        for p in (result.get("top_publications") or [])[:req.top_pubs]:
            srcs.append(Hit(id=p.get("title"), title=p.get("title"), venue=p.get("journal"), article_link=p.get("article_link")))
        return ChatResponse(route="A", answer="\n".join(ans), data=result, sources=srcs)


    # Route B (Hybrid retrieval + DeepSeek-R1)
    rag_res = rag(RagRequest(
        query=req.query,
        k=req.k,
        k_dense=req.k_dense,
        k_lex=req.k_lex,
        k_rrf=req.k_rrf,
        fused_threshold=req.fused_threshold,  
    ))

    fused_hits = rag_res.fused_top  

    # If nothing passes the threshold, short-circuit
    if not fused_hits:
        return ChatResponse(
            route="B",
            answer="اطلاعات کافی نیست.",
            data={"query": req.query, "note": "no sources under fused_threshold"},
            sources=[],
        )

    # Keep LLM context compact (but return all sources to the client)
    ctx_hits = fused_hits[:24]
    ctx = _ctx_from_hits(ctx_hits)
    prompt = (
        f"پرسش: {req.query}\n\n"
        f"منابع:\n{ctx}\n\n"
        "دستور پاسخ‌دهی:\n"
        "1) فقط از جملات موجود در منابع استفاده کن و بازنویسی کوتاه انجام بده.\n"
        "2) اگر پاسخ کامل نیست یا در منابع نیامده، بنویس: «اطلاعات کافی نیست».\n"
        "3) هر جملهٔ پاسخ را با ارجاع [شماره‌منبع] تمام کن. مثال: «… [1]»\n"
        "4) خیلی کوتاه و دقیق.\n"
        "5) خروجی را فقط به فارسی بنویس.\n"
    )

    answer = r1_generate(prompt, system=SYSTEM_MSG, stream=False)
    return ChatResponse(
        route="B",
        answer=answer.strip(),
        data={"query": req.query, "ctx_cap": 24, "fused_threshold": req.fused_threshold},
        sources=fused_hits,  # FULL filtered list
    )

# ---------- Endpoint Chat (streaming SSE) ----------
@app.options("/chat/stream")
def chat_stream_options():
    # CORS middleware will add the Access-Control-* headers
    return Response(status_code=204)

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    def gen():
        # Route with Block 4
        intent, name = R.extract_track_a(req.query)

        # ---------- Track A (deterministic) ----------
        if intent is not None:
            # same resolution logic as /chat
            if not name:
                yield _sse_pack({"type": "route", "route": "clarify"})
                yield _sse_pack({"type": "token", "delta":
                    "نام کامل استاد را دقیق وارد کنید (نام + نام خانوادگی) تا نتیجهٔ درخواستی ارائه شود.\n"})
                yield _sse_pack({"type": "done"})
                return

            rec = A_INDEX.find_exact(name) if A_INDEX else None
            if rec is None:
                resolved = _resolve_prof_name_flex(name)
                if resolved:
                    rec = A_INDEX.find_exact(resolved)

            if rec is None:
                # list candidates if any
                qn = _norm(name)
                candidates = []
                for n_norm, disp in PROF_NORM_TO_DISPLAY.items():
                    n_tokens = n_norm.split()
                    if qn in n_norm or qn in n_tokens:
                        candidates.append(disp)
                yield _sse_pack({"type": "route", "route": "clarify"})
                if candidates:
                    yield _sse_pack({"type": "token", "delta": "چند استاد با این نام پیدا شد؛ یکی را انتخاب کنید:\n"})
                    yield _sse_pack({"type": "candidates", "candidates": sorted(set(candidates))[:7]})
                else:
                    yield _sse_pack({"type": "token", "delta": "هیچ نام دقیقی پیدا نشد؛ لطفاً نام کامل فارسی استاد را بدهید.\n"})
                yield _sse_pack({"type": "done"})
                return

            # we have a record
            result = A.format_profile_output(rec, top=req.top_pubs, only="all")
            ans = []
            if intent == "email" and result.get("email"):
                ans.append(f"ایمیل {result['name']}: {result['email']}")
            elif intent == "homepage" and result.get("homepage"):
                ans.append(f"صفحهٔ شخصی {result['name']}: {result['homepage']}")
            else:
                pubs = result.get("top_publications") or []
                ans.append(f"۵ مورد از مقالات {result['name']} (در صورت وجود لینک، در اولویت):")
                for i, p in enumerate(pubs[:req.top_pubs], start=1):
                    t = p.get("title") or ""
                    v = p.get("journal") or ""
                    link = p.get("article_link") or ""
                    ans.append(f"{i}. {t} — {v} {'('+link+')' if link else ''}")

            # stream route + sources + answer text
            yield _sse_pack({"type": "route", "route": "A"})
            srcs: List[Hit] = []
            if result.get("homepage"):
                srcs.append(Hit(id="homepage", title="صفحه شخصی", venue=result["name"], article_link=result["homepage"]))
            for p in (result.get("top_publications") or [])[:req.top_pubs]:
                srcs.append(Hit(id=p.get("title"), title=p.get("title"), venue=p.get("journal"), article_link=p.get("article_link")))
            yield _sse_pack({"type": "sources", "sources": [s.dict() for s in srcs]})

            yield from _stream_deterministic_answer("\n".join(ans))
            return

        # ---------- Track B (RAG + LLM streaming) ----------
        rag_res = rag(RagRequest(
            query=req.query,
            k=req.k,
            k_dense=req.k_dense,
            k_lex=req.k_lex,
            k_rrf=req.k_rrf,
            fused_threshold=req.fused_threshold,  
        ))
        fused_hits = rag_res.fused_top  

        yield _sse_pack({"type": "route", "route": "B"})
        yield _sse_pack({"type": "sources", "sources": [h.dict() for h in fused_hits]})

        if not fused_hits:
            yield _sse_pack({"type": "token", "delta": "اطلاعات کافی نیست.\n"})
            yield _sse_pack({"type": "done"})
            return

        ctx_hits = fused_hits[:24]  # cap prompt size
        ctx = _ctx_from_hits(ctx_hits)
        prompt = (
            f"پرسش: {req.query}\n\n"
            f"منابع:\n{ctx}\n\n"
            "دستور پاسخ‌دهی:\n"
            "1) فقط از جملات موجود در منابع استفاده کن و بازنویسی کوتاه انجام بده.\n"
            "2) اگر پاسخ کامل نیست یا در منابع نیامده، بنویس: «اطلاعات کافی نیست».\n"
            "3) هر جملهٔ پاسخ را با ارجاع [شماره‌منبع] تمام کن. مثال: «… [1]»\n"
            "4) خیلی کوتاه و دقیق.\n"
            "5) خروجی را فقط به فارسی بنویس.\n"
        )

        for chunk in r1_generate_stream(prompt, system=SYSTEM_MSG):
            yield _sse_pack({"type": "token", "delta": chunk})
        yield _sse_pack({"type": "done"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )
