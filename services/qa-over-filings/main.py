import os
import re
import time
import uuid
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from flask import Flask, request, jsonify
from google.cloud import spanner

# LLM (Vertex backend)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Embeddings (Vertex backend)
from google import genai

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ----------------------------
# Env
# ----------------------------
SPANNER_PROJECT = os.environ["SPANNER_PROJECT"]
SPANNER_INSTANCE = os.environ["SPANNER_INSTANCE"]
SPANNER_DATABASE = os.environ["SPANNER_DATABASE"]

CRON_SECRET = os.environ.get("CRON_SECRET", "")

# Vertex / model
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")

# Embeddings (must match your embed-chunks service + Spanner vector_length)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "3072"))  # gemini-embedding-001 commonly returns 3072
NUM_LEAVES_TO_SEARCH = int(os.environ.get("NUM_LEAVES_TO_SEARCH", "10"))

# Retrieval knobs
TOP_K = int(os.environ.get("TOP_K", "6"))
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "60000"))

app = Flask(__name__)

_spanner_db = None
_llm = None
_genai_client = None

CITE_RE = re.compile(r"\[C(\d+)\]")


# ----------------------------
# Clients
# ----------------------------
def spanner_db():
    global _spanner_db
    if _spanner_db is None:
        sp = spanner.Client(project=SPANNER_PROJECT)
        inst = sp.instance(SPANNER_INSTANCE)
        _spanner_db = inst.database(SPANNER_DATABASE)
    return _spanner_db


def spanner_query(sql: str, params=None, param_types=None):
    db = spanner_db()
    with db.snapshot() as snap:
        return list(snap.execute_sql(sql, params=params, param_types=param_types))


def llm() -> ChatGoogleGenerativeAI:
    """
    Uses langchain-google-genai with Vertex AI backend.
    Requires ADC locally or service account on Cloud Run.
    """
    global _llm
    if _llm is None:
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", SPANNER_PROJECT)
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", VERTEX_LOCATION)

        _llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            project=SPANNER_PROJECT,
            location=VERTEX_LOCATION,
            vertexai=True,
            temperature=0.2,
        )
    return _llm


def genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", SPANNER_PROJECT)
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", VERTEX_LOCATION)

        _genai_client = genai.Client(
            vertexai=True,
            project=SPANNER_PROJECT,
            location=VERTEX_LOCATION,
        )
    return _genai_client


# ----------------------------
# Filing selection
# ----------------------------
def get_latest_accession_for_form(ticker: str, form_type: str) -> Optional[str]:
    sql = """
      SELECT AccessionNo
      FROM Filing
      WHERE Ticker=@t AND FormType=@f AND Status IN ("CHUNKED","EMBEDDED")
      ORDER BY FiledAt DESC
      LIMIT 1
    """
    params = {"t": ticker, "f": form_type}
    ptypes = {"t": spanner.param_types.STRING, "f": spanner.param_types.STRING}
    rows = spanner_query(sql, params=params, param_types=ptypes)
    return rows[0][0] if rows else None


# ----------------------------
# Retrieval: Vector (preferred) + LIKE fallback
# ----------------------------
def embed_query_text(text: str) -> List[float]:
    """
    Returns an embedding vector for the user query (float list length EMBED_DIM).
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty query text")

    resp = genai_client().models.embed_content(
        model=EMBED_MODEL,
        contents=[text],
    )
    vec = [float(x) for x in resp.embeddings[0].values]
    if len(vec) != EMBED_DIM:
        raise ValueError(f"query embedding dim={len(vec)} != EMBED_DIM={EMBED_DIM}")
    return vec


def retrieve_chunks_vector(
    ticker: str, accession: str, query_vec: List[float], top_k: int
) -> List[Dict[str, Any]]:
    """
    ANN vector search using FilingChunkEmbeddingIdx.
    Requires:
      - FilingChunk.Embedding populated (NOT NULL)
      - VECTOR INDEX FilingChunkEmbeddingIdx exists
    """
    sql = """
      SELECT ChunkId, Section, ChunkText, StartOffset, EndOffset,
        APPROX_COSINE_DISTANCE(
          Embedding,
          @q,
          options => JSON @opts
        ) AS distance
      FROM FilingChunk @{force_index=FilingChunkEmbeddingIdx}
      WHERE Ticker=@t AND AccessionNo=@a AND Embedding IS NOT NULL
      ORDER BY distance
      LIMIT @k
    """
    params = {
        "t": ticker,
        "a": accession,
        "q": query_vec,
        "k": top_k,
        "opts": json.dumps({"num_leaves_to_search": NUM_LEAVES_TO_SEARCH}),
    }
    ptypes = {
        "t": spanner.param_types.STRING,
        "a": spanner.param_types.STRING,
        "k": spanner.param_types.INT64,
        "opts": spanner.param_types.STRING,
        # NOTE: We intentionally omit the param type for @q because FLOAT32 array param typing
        # can vary by client version; the Spanner client will usually infer it correctly.
        # If your client errors, set:
        # "q": spanner.param_types.Array(spanner.param_types.FLOAT32)
    }

    rows = spanner_query(sql, params=params, param_types=ptypes)
    out = []
    for r in rows:
        out.append(
            {
                "chunk_id": r[0],
                "section": r[1],
                "text": r[2],
                "start": r[3],
                "end": r[4],
                "score": float(r[5]) if r[5] is not None else None,  # distance
            }
        )
    return out


def retrieve_chunks_like(ticker: str, accession: str, query_text: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Simple lexical LIKE retrieval (fallback).
    """
    words = [w.strip().lower() for w in (query_text or "").split() if len(w.strip()) >= 3][:12]

    if not words:
        sql = """
          SELECT ChunkId, Section, ChunkText, StartOffset, EndOffset
          FROM FilingChunk
          WHERE Ticker=@t AND AccessionNo=@a
          ORDER BY ChunkId
          LIMIT @k
        """
        params = {"t": ticker, "a": accession, "k": top_k}
        ptypes = {
            "t": spanner.param_types.STRING,
            "a": spanner.param_types.STRING,
            "k": spanner.param_types.INT64,
        }
        rows = spanner_query(sql, params=params, param_types=ptypes)
        return [{"chunk_id": r[0], "section": r[1], "text": r[2], "start": r[3], "end": r[4], "score": None} for r in rows]

    like_clauses = []
    params = {"t": ticker, "a": accession, "k": top_k}
    ptypes = {
        "t": spanner.param_types.STRING,
        "a": spanner.param_types.STRING,
        "k": spanner.param_types.INT64,
    }

    for i, w in enumerate(words):
        pname = f"w{i}"
        like_clauses.append(f"LOWER(ChunkText) LIKE @{pname}")
        params[pname] = f"%{w}%"
        ptypes[pname] = spanner.param_types.STRING

    sql = f"""
      SELECT ChunkId, Section, ChunkText, StartOffset, EndOffset
      FROM FilingChunk
      WHERE Ticker=@t AND AccessionNo=@a
        AND ({' OR '.join(like_clauses)})
      ORDER BY ChunkId
      LIMIT @k
    """
    rows = spanner_query(sql, params=params, param_types=ptypes)
    return [{"chunk_id": r[0], "section": r[1], "text": r[2], "start": r[3], "end": r[4], "score": None} for r in rows]


def retrieve_chunks(ticker: str, accession: str, query_text: str, top_k: int) -> Dict[str, Any]:
    """
    Prefer vector search when possible; fallback to LIKE when embeddings are missing or
    vector query fails for any reason.
    Returns: { mode, chunks, vector_error }
    """
    vector_error = None

    try:
        qvec = embed_query_text(query_text)
        chunks = retrieve_chunks_vector(ticker, accession, qvec, top_k)
        if chunks:
            return {"mode": "vector", "chunks": chunks, "vector_error": None}
        # If vector returns nothing, we still fallback to LIKE.
    except Exception as e:
        vector_error = str(e)

    chunks = retrieve_chunks_like(ticker, accession, query_text, top_k)
    return {"mode": "like", "chunks": chunks, "vector_error": vector_error}


def build_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Build context blocks with citations [C1], [C2], ...
    """
    parts = []
    total = 0
    for i, c in enumerate(chunks, start=1):
        tag = f"[C{i}]"
        score = c.get("score")
        score_txt = f" distance={score:.6f}" if isinstance(score, float) else ""
        block = f"{tag} ChunkId={c['chunk_id']} Section={c.get('section')}{score_txt}\n{c['text']}\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n---\n".join(parts)


# ----------------------------
# LLM prompt (natural language output)
# ----------------------------
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a financial filings analyst. Answer ONLY using the provided context.\n"
            "If the answer is not in the context, say you don't know.\n"
            "\n"
            "Write a DETAILED answer (aim for 6–12 sentences) and include:\n"
            "1) a 1–2 sentence summary\n"
            "2) 3–6 bullet points of key facts\n"
            "3) what the filing is / why it was filed (if stated)\n"
            "\n"
            "Citations: add citations like [C1], [C2] at the end of EACH sentence or bullet it supports.\n"
            "Return plain text (no JSON, no markdown/code fences)."
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)


def extract_citations_used(answer_text: str) -> List[str]:
    seen = set()
    for m in CITE_RE.finditer(answer_text or ""):
        seen.add(f"C{m.group(1)}")
    return sorted(seen, key=lambda s: int(s[1:]))


def ask_llm(question: str, context: str) -> str:
    chain = PROMPT | llm()
    resp = chain.invoke({"question": question, "context": context})
    return (resp.content if hasattr(resp, "content") else str(resp)).strip()


# ----------------------------
# API
# ----------------------------
@app.route("/ask", methods=["POST"])
def ask():
    rid = uuid.uuid4().hex[:8]
    t0 = time.time()

    if CRON_SECRET:
        hdr = request.headers.get("X-Cron-Secret", "")
        if hdr != CRON_SECRET:
            return jsonify({"error": "forbidden"}), 403

    body = request.get_json(silent=True) or {}
    ticker = (body.get("ticker") or "").strip().upper()
    question = (body.get("question") or "").strip()
    form_type = (body.get("form_type") or "10-K").strip().upper()
    accession = (body.get("accession") or "").strip()

    if not ticker or not question:
        return jsonify({"error": "ticker and question are required"}), 400

    if not accession:
        accession = get_latest_accession_for_form(ticker, form_type)
        if not accession:
            return jsonify({"error": f"no CHUNKED/EMBEDDED filings found for {ticker} {form_type}"}), 404

    log.info("[%s] ask: %s %s accession=%s", rid, ticker, form_type, accession)

    retr = retrieve_chunks(ticker, accession, question, TOP_K)
    mode = retr["mode"]
    chunks = retr["chunks"]
    vector_error = retr.get("vector_error")

    context = build_context(chunks)

    try:
        answer = ask_llm(question, context)
    except Exception as e:
        log.exception("[%s] LLM call failed: %s", rid, e)
        return jsonify(
            {
                "error": str(e),
                "hint": "Vertex backend: ensure VERTEX_LOCATION=global and the model is enabled for your project.",
                "request_id": rid,
            }
        ), 500

    citations = []
    for i, c in enumerate(chunks, start=1):
        citations.append(
            {
                "cite": f"[C{i}]",
                "ticker": ticker,
                "accession": accession,
                "chunk_id": c["chunk_id"],
                "section": c.get("section"),
                "start_offset": c.get("start"),
                "end_offset": c.get("end"),
                "score": c.get("score"),
            }
        )

    result = {
        "request_id": rid,
        "utc": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "form_type": form_type,
        "accession": accession,
        "top_k": TOP_K,
        "retrieval_mode": mode,
        "vector_error": vector_error,
        "answer": answer,
        "citations": citations,
        "citations_used": extract_citations_used(answer),
        "seconds": round(time.time() - t0, 3),
    }
    return jsonify(result), 200


@app.route("/", methods=["GET"])
def root():
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=True)
