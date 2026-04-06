import os
import re
import time
import uuid
import json
import logging
import ssl
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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
SPANNER_PROJECT = os.environ.get("SPANNER_PROJECT", "")
SPANNER_INSTANCE = os.environ.get("SPANNER_INSTANCE", "")
SPANNER_DATABASE = os.environ.get("SPANNER_DATABASE", "")

CRON_SECRET = os.environ.get("CRON_SECRET", "")

# Vertex / model
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")

# Embeddings (must match your embed-chunks service + Spanner vector_length)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))
NUM_LEAVES_TO_SEARCH = int(os.environ.get("NUM_LEAVES_TO_SEARCH", "10"))

# Retrieval knobs
TOP_K = int(os.environ.get("TOP_K", "6"))
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "60000"))

# Summary knobs
SUMMARY_LOOKBACK_DAYS = int(os.environ.get("SUMMARY_LOOKBACK_DAYS", "7"))
DAILY_LOOKBACK_DAYS = int(os.environ.get("DAILY_LOOKBACK_DAYS", "1"))

SUMMARY_MAX_FILINGS_PER_COMPANY = int(os.environ.get("SUMMARY_MAX_FILINGS_PER_COMPANY", "5"))
SUMMARY_MAX_COMPANIES = int(os.environ.get("SUMMARY_MAX_COMPANIES", "25"))
SUMMARY_MAX_CHUNKS_PER_FILING = int(os.environ.get("SUMMARY_MAX_CHUNKS_PER_FILING", "8"))
SUMMARY_MAX_CHARS_PER_FILING = int(os.environ.get("SUMMARY_MAX_CHARS_PER_FILING", "12000"))

SUMMARY_FORM_TYPES_ALLOW = os.environ.get(
    "SUMMARY_FORM_TYPES_ALLOW",
    "8-K,10-Q,10-K,6-K,S-1,424B,424B1,424B2,424B3,424B4,424B5,424B7,424B8",
)
SUMMARY_FORM_TYPES_DENY = os.environ.get("SUMMARY_FORM_TYPES_DENY", "3,4,5,SC 13D,SC 13G")

# ----------------------------
# Email delivery (Gmail SMTP + HTML)
# ----------------------------
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")          # e.g. jenniferwang1007@gmail.com
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")  # mounted from Secret Manager (App Password)

EMAIL_FROM = os.environ.get("EMAIL_FROM", "")        # should match SMTP_USER for Gmail
EMAIL_TO = os.environ.get("EMAIL_TO", "")
EMAIL_SUBJECT_PREFIX = os.environ.get("EMAIL_SUBJECT_PREFIX", "SEC Filing Impact Summary")

app = Flask(__name__)

_spanner_db = None
_llm = None
_genai_client = None

CITE_RE = re.compile(r"\[C(\d+)\]")


# ----------------------------
# Helpers
# ----------------------------
def _require_env():
    missing = []
    if not SPANNER_PROJECT:
        missing.append("SPANNER_PROJECT")
    if not SPANNER_INSTANCE:
        missing.append("SPANNER_INSTANCE")
    if not SPANNER_DATABASE:
        missing.append("SPANNER_DATABASE")
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


def _parse_csv_list(s: str) -> List[str]:
    out = []
    for x in (s or "").split(","):
        x = x.strip()
        if x:
            out.append(x.upper())
    return out


ALLOW_FORMS = set(_parse_csv_list(SUMMARY_FORM_TYPES_ALLOW))
DENY_FORMS = set(_parse_csv_list(SUMMARY_FORM_TYPES_DENY))


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_citations_used(answer_text: str) -> List[str]:
    seen = set()
    for m in CITE_RE.finditer(answer_text or ""):
        seen.add(f"C{m.group(1)}")
    return sorted(seen, key=lambda s: int(s[1:]))


def strip_citations(text: str) -> str:
    # Removes [C1], [C2]... but keeps formatting like **bold**
    return re.sub(r"\s*\[C\d+\]\s*", " ", text or "").strip()


def extract_company_name_from_chunks(chunks: List[Dict[str, Any]]) -> str:
    """
    Extract the registrant name from filing text (grounded).
    Common patterns seen in SEC .txt/XBRL cover blocks.
    """
    patterns = [
        r"Entity Registrant Name\s+([A-Za-z0-9.,&()\-/' ]+)",
        r"Registrant Name\s*[:\-]\s*([A-Za-z0-9.,&()\-/' ]+)",
        r"Company Conformed Name\s*[:\-]\s*([A-Za-z0-9.,&()\-/' ]+)",
    ]
    for c in chunks:
        text = c.get("text") or ""
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
    return ""


# ----------------------------
# Email helpers (Gmail SMTP)
# ----------------------------
def _email_require_env():
    missing = []
    if not SMTP_USER:
        missing.append("SMTP_USER")
    if not SMTP_PASSWORD:
        missing.append("SMTP_PASSWORD")
    if not EMAIL_FROM:
        missing.append("EMAIL_FROM")
    if not EMAIL_TO:
        missing.append("EMAIL_TO")
    if missing:
        raise RuntimeError(f"Missing required email env vars: {', '.join(missing)}")


def _plain_to_html(body_text: str) -> str:
    """
    Convert your existing preview text into a simple HTML email.
    - Converts **bold** to <strong>
    - Converts newlines to <br>
    - Escapes HTML special chars first (so filings text doesn't break HTML)
    """
    s = body_text or ""
    # escape first
    s = (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )
    # convert **bold**
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    # keep indentation readable
    s = s.replace("\n", "<br>\n")
    return f"""\
<html>
  <body style="font-family: Arial, Helvetica, sans-serif; font-size: 14px; line-height: 1.4;">
    {s}
  </body>
</html>
"""


def send_email_gmail_smtp(subject: str, body_text: str) -> Dict[str, Any]:
    """
    Sends an HTML email via Gmail SMTP using an App Password.
    """
    _email_require_env()

    html = _plain_to_html(body_text)

    msg = MIMEMultipart("alternative")
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject

    # Add both plain + html (gmail clients pick html; others fallback to plain)
    msg.attach(MIMEText(body_text or "", "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))

    context = ssl.create_default_context()

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            try:
                server.login(SMTP_USER, SMTP_PASSWORD)
            except smtplib.SMTPAuthenticationError as e:
                return {"ok": False, "status": 401, "error": f"SMTP auth failed: {e}"}

            server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())
        return {"ok": True, "status": 200}
    except smtplib.SMTPException as e:
        return {"ok": False, "status": 500, "error": f"SMTP error: {e}"}
    except Exception as e:
        return {"ok": False, "status": 500, "error": str(e)}


# ----------------------------
# Clients
# ----------------------------
def spanner_db():
    global _spanner_db
    if _spanner_db is None:
        _require_env()
        sp = spanner.Client(project=SPANNER_PROJECT)
        inst = sp.instance(SPANNER_INSTANCE)
        _spanner_db = inst.database(SPANNER_DATABASE)
    return _spanner_db


def spanner_query(sql: str, params=None, param_types=None):
    db = spanner_db()
    with db.snapshot() as snap:
        return list(snap.execute_sql(sql, params=params, param_types=param_types))


def llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _require_env()
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
        _require_env()
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
def get_latest_accession_for_form(ticker: str, form_type: str, lookback_days: int) -> Optional[str]:
    sql = """
      SELECT AccessionNo
      FROM Filing
      WHERE Ticker=@t
        AND FormType=@f
        AND Status IN ("CHUNKED","EMBEDDED")
        AND FiledAt >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @d DAY)
      ORDER BY FiledAt DESC
      LIMIT 1
    """
    params = {"t": ticker, "f": form_type, "d": lookback_days}
    ptypes = {
        "t": spanner.param_types.STRING,
        "f": spanner.param_types.STRING,
        "d": spanner.param_types.INT64,
    }
    rows = spanner_query(sql, params=params, param_types=ptypes)
    return rows[0][0] if rows else None


def list_recent_filings(
    ticker: Optional[str],
    lookback_days: int,
    max_companies: int,
    max_filings_per_company: int,
) -> List[Dict[str, Any]]:
    """
    Returns filings from last N days, optionally filtered to a ticker.
    Applies allow/deny on FormType when allow list is non-empty.
    """
    sql = """
      SELECT Ticker, AccessionNo, FormType, FiledAt
      FROM Filing
      WHERE Status IN ("CHUNKED","EMBEDDED")
        AND FiledAt >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @d DAY)
        AND (@t = "" OR Ticker=@t)
      ORDER BY FiledAt DESC
      LIMIT 2000
    """
    params = {"d": lookback_days, "t": (ticker or "").upper()}
    ptypes = {"d": spanner.param_types.INT64, "t": spanner.param_types.STRING}
    rows = spanner_query(sql, params=params, param_types=ptypes)

    out: List[Dict[str, Any]] = []
    per_ticker: Dict[str, int] = {}
    seen_tickers: List[str] = []

    for r in rows:
        tkr = (r[0] or "").upper()
        acc = r[1]
        form = (r[2] or "").upper()
        filed_at = r[3]

        if not tkr or not acc or not form:
            continue

        if ALLOW_FORMS and form not in ALLOW_FORMS:
            continue
        if form in DENY_FORMS:
            continue

        if tkr not in per_ticker:
            per_ticker[tkr] = 0
            seen_tickers.append(tkr)
            if ticker is None and len(seen_tickers) > max_companies:
                continue

        if per_ticker[tkr] >= max_filings_per_company:
            continue

        per_ticker[tkr] += 1
        out.append({"ticker": tkr, "accession": acc, "form_type": form, "filed_at": filed_at})

    return out


# ----------------------------
# Retrieval: Vector (preferred) + LIKE fallback
# ----------------------------
def embed_query_text(text: str) -> List[float]:
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
    opts_literal = f'{{"num_leaves_to_search": {NUM_LEAVES_TO_SEARCH}}}'

    sql = f"""
      SELECT ChunkId, Section, ChunkText, StartOffset, EndOffset,
        APPROX_COSINE_DISTANCE(
          Embedding,
          @q,
          options => JSON '{opts_literal}'
        ) AS distance
      FROM FilingChunk @{{force_index=FilingChunkEmbeddingIdx}}
      WHERE Ticker=@t AND AccessionNo=@a AND Embedding IS NOT NULL
      ORDER BY distance
      LIMIT @k
    """
    params = {"t": ticker, "a": accession, "q": query_vec, "k": top_k}
    ptypes = {
        "t": spanner.param_types.STRING,
        "a": spanner.param_types.STRING,
        "k": spanner.param_types.INT64,
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
                "score": float(r[5]) if r[5] is not None else None,
            }
        )
    return out


def retrieve_chunks_like(
    ticker: str, accession: str, query_text: str, top_k: int
) -> List[Dict[str, Any]]:
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
        return [
            {"chunk_id": r[0], "section": r[1], "text": r[2], "start": r[3], "end": r[4], "score": None}
            for r in rows
        ]

    like_clauses = []
    params = {"t": ticker, "a": accession, "k": top_k}
    ptypes = {"t": spanner.param_types.STRING, "a": spanner.param_types.STRING, "k": spanner.param_types.INT64}

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
    return [
        {"chunk_id": r[0], "section": r[1], "text": r[2], "start": r[3], "end": r[4], "score": None}
        for r in rows
    ]


def retrieve_chunks(ticker: str, accession: str, query_text: str, top_k: int) -> Dict[str, Any]:
    vector_error = None
    try:
        qvec = embed_query_text(query_text)
        chunks = retrieve_chunks_vector(ticker, accession, qvec, top_k)
        if chunks:
            return {"mode": "vector", "chunks": chunks, "vector_error": None}
    except Exception as e:
        vector_error = str(e)

    chunks = retrieve_chunks_like(ticker, accession, query_text, top_k)
    return {"mode": "like", "chunks": chunks, "vector_error": vector_error}


def build_context(chunks: List[Dict[str, Any]], max_chars: int) -> str:
    parts = []
    total = 0
    for i, c in enumerate(chunks, start=1):
        tag = f"[C{i}]"
        score = c.get("score")
        score_txt = f" distance={score:.6f}" if isinstance(score, float) else ""
        block = f"{tag} ChunkId={c['chunk_id']} Section={c.get('section')}{score_txt}\n{c['text']}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n---\n".join(parts)


# ----------------------------
# LLM prompts
# ----------------------------
ASK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a financial filings analyst. Answer ONLY using the provided context.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            "Write a DETAILED answer (aim for 6–12 sentences) and include:\n"
            "1) a 1–2 sentence summary\n"
            "2) 3–6 bullet points of key facts\n"
            "3) what the filing is / why it was filed (if stated)\n\n"
            "Citations: add citations like [C1], [C2] at the end of EACH sentence or bullet it supports.\n"
            "Return plain text (no JSON, no markdown/code fences)."
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)

# Force company name in every sentence (no "This filing..." without naming)
FILING_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an investor-focused SEC filings analyst.\n"
            "Use ONLY the provided context.\n"
            "Do NOT guess the company name.\n\n"
            "You MUST explicitly include the company name in EVERY sentence.\n"
            "Do NOT use phrases like 'This filing' without naming the company.\n\n"
            "You MUST use the given company_name (extracted from filing text).\n"
            "If company_name indicates it was not found, say that explicitly.\n\n"
            "Required output format (plain text):\n"
            "**{company_name}**\n"
            "Summary:\n"
            "- Start the sentence with {company_name} and describe what the {form_type} contains/is about.\n"
            "Stock Impact:\n"
            "- Neutral / Bullish / Bearish — one-line reason that ALSO includes {company_name}.\n\n"
            "End each bullet with citations like [C1].\n"
            "No markdown fences."
        ),
        ("human", "Filing: {form_type} {accession}\n\nContext:\n{context}"),
    ]
)


def ask_llm(prompt: ChatPromptTemplate, variables: Dict[str, Any]) -> str:
    chain = prompt | llm()
    resp = chain.invoke(variables)
    return (resp.content if hasattr(resp, "content") else str(resp)).strip()


# ----------------------------
# Auth
# ----------------------------
def _check_cron_secret() -> Optional[Tuple[Dict[str, Any], int]]:
    if CRON_SECRET:
        hdr = request.headers.get("X-Cron-Secret", "")
        if hdr != CRON_SECRET:
            return {"error": "forbidden"}, 403
    return None


# ----------------------------
# API: /ask (single filing QA)
# ----------------------------
@app.route("/ask", methods=["POST"])
def ask():
    rid = uuid.uuid4().hex[:8]
    t0 = time.time()

    forbid = _check_cron_secret()
    if forbid:
        return jsonify(forbid[0]), forbid[1]

    body = request.get_json(silent=True) or {}
    ticker = (body.get("ticker") or "").strip().upper()
    question = (body.get("question") or "").strip()
    form_type = (body.get("form_type") or "10-K").strip().upper()
    accession = (body.get("accession") or "").strip()
    lookback_days = int(body.get("lookback_days") or SUMMARY_LOOKBACK_DAYS)

    if not ticker or not question:
        return jsonify({"error": "ticker and question are required"}), 400

    if not accession:
        accession = get_latest_accession_for_form(ticker, form_type, lookback_days)
        if not accession:
            return jsonify({"error": f"no CHUNKED/EMBEDDED filings found for {ticker} {form_type}"}), 404

    log.info("[%s] ask: %s %s accession=%s", rid, ticker, form_type, accession)

    retr = retrieve_chunks(ticker, accession, question, TOP_K)
    chunks = retr["chunks"]
    context = build_context(chunks, MAX_CONTEXT_CHARS)

    try:
        answer = ask_llm(ASK_PROMPT, {"question": question, "context": context})
    except Exception as e:
        log.exception("[%s] LLM call failed: %s", rid, e)
        return jsonify({"error": str(e), "request_id": rid}), 500

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

    return (
        jsonify(
            {
                "request_id": rid,
                "utc": _now_utc_iso(),
                "ticker": ticker,
                "form_type": form_type,
                "accession": accession,
                "top_k": TOP_K,
                "retrieval_mode": retr["mode"],
                "vector_error": retr.get("vector_error"),
                "answer": answer,
                "citations": citations,
                "citations_used": extract_citations_used(answer),
                "seconds": round(time.time() - t0, 3),
            }
        ),
        200,
    )


# ----------------------------
# Core summary generator (used by weekly/daily + email)
# ----------------------------
def generate_summary_blocks(lookback_days: int, run_utc: str, ticker: str = "") -> Dict[str, Any]:
    rid = uuid.uuid4().hex[:8]
    t0 = time.time()

    ticker_filter: Optional[str] = ticker.strip().upper() if ticker else None

    filings = list_recent_filings(
        ticker=ticker_filter,
        lookback_days=lookback_days,
        max_companies=SUMMARY_MAX_COMPANIES,
        max_filings_per_company=SUMMARY_MAX_FILINGS_PER_COMPANY,
    )

    if not filings:
        msg = "no recent CHUNKED/EMBEDDED filings found"
        if ticker_filter:
            msg += f" for {ticker_filter}"
        return {"error": msg, "request_id": rid, "utc": _now_utc_iso()}

    summary_query = (
        "Find the registrant name and summarize any material disclosures, events, numbers, guidance, "
        "risk factors, financing, M&A, legal/regulatory items, or anything likely to move the stock."
    )

    per_company: Dict[str, List[Dict[str, Any]]] = {}
    per_filing_outputs: List[Dict[str, Any]] = []

    for f in filings:
        tkr = f["ticker"]
        acc = f["accession"]
        form = f["form_type"]
        filed_at = f["filed_at"]

        retr = retrieve_chunks(tkr, acc, summary_query, SUMMARY_MAX_CHUNKS_PER_FILING)
        chunks = retr["chunks"]
        context = build_context(chunks, SUMMARY_MAX_CHARS_PER_FILING)

        company_name = extract_company_name_from_chunks(chunks)
        if not company_name:
            company_name = f"{tkr} (registrant name not found in retrieved text)"

        try:
            block = ask_llm(
                FILING_SUMMARY_PROMPT,
                {"company_name": company_name, "form_type": form, "accession": acc, "context": context},
            )
        except Exception as e:
            log.exception("[%s] per-filing summary failed (%s %s): %s", rid, tkr, acc, e)
            block = (
                f"**{company_name}**\n"
                f"Summary:\n"
                f"- {company_name} could not be summarized due to an internal error. [C1]\n"
                f"Stock Impact:\n"
                f"- Neutral — {company_name} analysis failed due to an internal error. [C1]"
            )

        item = {
            "ticker": tkr,
            "company_name": company_name,
            "form_type": form,
            "accession": acc,
            "filed_at": str(filed_at),
            "retrieval_mode": retr["mode"],
            "vector_error": retr.get("vector_error"),
            "block": block,
            "citations_used": extract_citations_used(block),
        }
        per_filing_outputs.append(item)
        per_company.setdefault(tkr, []).append(item)

    # Stable ordering: ticker asc; per-ticker insertion order is newest-first from query
    blocks_parts: List[str] = []
    for tkr in sorted(per_company.keys()):
        for it in per_company[tkr]:
            blocks_parts.append(it["block"].strip())
            blocks_parts.append("")
    blocks_blob = "\n".join(blocks_parts).strip()

    # Return ONLY company blocks, and strip citations (keep **bold** markers)
    blocks_only = strip_citations(blocks_blob)

    return {
        "request_id": rid,
        "utc": _now_utc_iso(),
        "run_utc": run_utc,
        "lookback_days": lookback_days,
        "ticker": ticker_filter or "",
        "filings_count": len(per_filing_outputs),
        "companies_count": len(per_company.keys()),
        "companies": [{"ticker": t, "filings": per_company[t]} for t in sorted(per_company.keys())],
        "email_preview": blocks_only,  # kept for compatibility with your jq commands
        "seconds": round(time.time() - t0, 3),
    }


# ----------------------------
# API: /weekly_summary (no email)
# ----------------------------
@app.route("/weekly_summary", methods=["POST"])
def weekly_summary():
    forbid = _check_cron_secret()
    if forbid:
        return jsonify(forbid[0]), forbid[1]

    body = request.get_json(silent=True) or {}
    ticker = (body.get("ticker") or "").strip().upper()
    lookback_days = int(body.get("lookback_days") or SUMMARY_LOOKBACK_DAYS)
    run_utc = (body.get("run_utc") or _now_utc_iso()).strip()

    resp = generate_summary_blocks(lookback_days=lookback_days, run_utc=run_utc, ticker=ticker)
    status = 200 if "error" not in resp else 404
    return jsonify(resp), status


# ----------------------------
# API: /daily_summary (no email)
# ----------------------------
@app.route("/daily_summary", methods=["POST"])
def daily_summary():
    forbid = _check_cron_secret()
    if forbid:
        return jsonify(forbid[0]), forbid[1]

    body = request.get_json(silent=True) or {}
    ticker = (body.get("ticker") or "").strip().upper()
    lookback_days = int(body.get("lookback_days") or DAILY_LOOKBACK_DAYS)
    run_utc = (body.get("run_utc") or _now_utc_iso()).strip()

    resp = generate_summary_blocks(lookback_days=lookback_days, run_utc=run_utc, ticker=ticker)
    status = 200 if "error" not in resp else 404
    return jsonify(resp), status


# ----------------------------
# API: /weekly_email (generate + send)
# ----------------------------
@app.route("/weekly_email", methods=["POST"])
def weekly_email():
    forbid = _check_cron_secret()
    if forbid:
        return jsonify(forbid[0]), forbid[1]

    body = request.get_json(silent=True) or {}
    ticker = (body.get("ticker") or "").strip().upper()
    lookback_days = int(body.get("lookback_days") or SUMMARY_LOOKBACK_DAYS)
    run_utc = (body.get("run_utc") or _now_utc_iso()).strip()

    resp = generate_summary_blocks(lookback_days=lookback_days, run_utc=run_utc, ticker=ticker)
    if "error" in resp:
        if "no recent CHUNKED/EMBEDDED filings found" in resp.get("error", ""):
            subject = f"{EMAIL_SUBJECT_PREFIX} (Weekly - {lookback_days}d)"
            if ticker:
                subject = f"{EMAIL_SUBJECT_PREFIX} ({ticker} - Weekly - {lookback_days}d)"
            body_text = "No new filings today."
            send_res = send_email_gmail_smtp(subject=subject, body_text=body_text)
            resp["email_sent"] = bool(send_res.get("ok"))
            resp["email_status"] = send_res.get("status")
            if not send_res.get("ok"):
                resp["email_error"] = send_res.get("error")
            return jsonify(resp), 200 if resp["email_sent"] else 502
        return jsonify(resp), 404

    subject = f"{EMAIL_SUBJECT_PREFIX} (Weekly - {lookback_days}d)"
    if ticker:
        subject = f"{EMAIL_SUBJECT_PREFIX} ({ticker} - Weekly - {lookback_days}d)"

    body_text = resp["email_preview"]

    send_res = send_email_gmail_smtp(subject=subject, body_text=body_text)
    resp["email_sent"] = bool(send_res.get("ok"))
    resp["email_status"] = send_res.get("status")
    if not send_res.get("ok"):
        resp["email_error"] = send_res.get("error")

    return jsonify(resp), 200 if resp["email_sent"] else 502


# ----------------------------
# API: /daily_email (generate + send)
# ----------------------------
@app.route("/daily_email", methods=["POST"])
def daily_email():
    forbid = _check_cron_secret()
    if forbid:
        return jsonify(forbid[0]), forbid[1]

    body = request.get_json(silent=True) or {}
    ticker = (body.get("ticker") or "").strip().upper()
    lookback_days = int(body.get("lookback_days") or DAILY_LOOKBACK_DAYS)
    run_utc = (body.get("run_utc") or _now_utc_iso()).strip()

    resp = generate_summary_blocks(lookback_days=lookback_days, run_utc=run_utc, ticker=ticker)
    if "error" in resp:
        if "no recent CHUNKED/EMBEDDED filings found" in resp.get("error", ""):
            subject = f"{EMAIL_SUBJECT_PREFIX} (Daily - {lookback_days}d)"
            if ticker:
                subject = f"{EMAIL_SUBJECT_PREFIX} ({ticker} - Daily - {lookback_days}d)"
            body_text = "No new filings today."
            send_res = send_email_gmail_smtp(subject=subject, body_text=body_text)
            resp["email_sent"] = bool(send_res.get("ok"))
            resp["email_status"] = send_res.get("status")
            if not send_res.get("ok"):
                resp["email_error"] = send_res.get("error")
            return jsonify(resp), 200 if resp["email_sent"] else 502
        return jsonify(resp), 404

    subject = f"{EMAIL_SUBJECT_PREFIX} (Daily - {lookback_days}d)"
    if ticker:
        subject = f"{EMAIL_SUBJECT_PREFIX} ({ticker} - Daily - {lookback_days}d)"

    body_text = resp["email_preview"]

    send_res = send_email_gmail_smtp(subject=subject, body_text=body_text)
    resp["email_sent"] = bool(send_res.get("ok"))
    resp["email_status"] = send_res.get("status")
    if not send_res.get("ok"):
        resp["email_error"] = send_res.get("error")

    return jsonify(resp), 200 if resp["email_sent"] else 502


@app.route("/", methods=["GET"])
def root():
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=True)