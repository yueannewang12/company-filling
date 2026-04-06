# embed-chunks/main.py
# Updated to support workflow-provided ticker override (request JSON)
# and remain fully idempotent.

import os
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
from collections import defaultdict

from flask import Flask, request, jsonify
from google.cloud import spanner
from google import genai

try:
    from google.genai import types as genai_types
except Exception:
    genai_types = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ----------------------------
# Env
# ----------------------------
SPANNER_PROJECT = os.environ["SPANNER_PROJECT"]
SPANNER_INSTANCE = os.environ["SPANNER_INSTANCE"]
SPANNER_DATABASE = os.environ["SPANNER_DATABASE"]

CRON_SECRET = os.environ.get("CRON_SECRET", "")

VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))

SPANNER_WRITE_BATCH = int(os.environ.get("SPANNER_WRITE_BATCH", "200"))
MAX_CHUNKS_PER_RUN = int(os.environ.get("MAX_CHUNKS_PER_RUN", "256"))
MAX_EMBED_CHARS = int(os.environ.get("MAX_EMBED_CHARS", "12000"))
SLEEP_BETWEEN_CALLS_SEC = float(os.environ.get("SLEEP_BETWEEN_CALLS_SEC", "0"))

UPDATE_FILING_STATUS = os.environ.get("UPDATE_FILING_STATUS", "true").lower() == "true"

app = Flask(__name__)

_spanner_db = None
_genai_client = None


# ----------------------------
# Clients
# ----------------------------
def spanner_db():
    global _spanner_db
    if _spanner_db is None:
        client = spanner.Client(project=SPANNER_PROJECT)
        inst = client.instance(SPANNER_INSTANCE)
        _spanner_db = inst.database(SPANNER_DATABASE)
    return _spanner_db


def spanner_query(sql, params=None, param_types=None):
    with spanner_db().snapshot() as snap:
        return list(snap.execute_sql(sql, params=params, param_types=param_types))


def genai_client():
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
# Spanner selection
# ----------------------------
def fetch_unembedded_chunks(limit: int, ticker: str | None):
    where = "Embedding IS NULL"
    params = {"lim": limit}
    ptypes = {"lim": spanner.param_types.INT64}

    if ticker:
        where += " AND Ticker=@t"
        params["t"] = ticker
        ptypes["t"] = spanner.param_types.STRING

    sql = f"""
      SELECT Ticker, AccessionNo, ChunkId, ChunkText
      FROM FilingChunk
      WHERE {where}
      ORDER BY Ticker, AccessionNo, ChunkId
      LIMIT @lim
    """
    rows = spanner_query(sql, params=params, param_types=ptypes)
    return [
        {
            "ticker": r[0],
            "accession": r[1],
            "chunk_id": int(r[2]),
            "text": r[3] or "",
        }
        for r in rows
    ]


def remaining_unembedded_count_for_filing(ticker: str, accession: str) -> int:
    sql = """
      SELECT COUNT(1)
      FROM FilingChunk
      WHERE Ticker=@t AND AccessionNo=@a AND Embedding IS NULL
    """
    params = {"t": ticker, "a": accession}
    ptypes = {"t": spanner.param_types.STRING, "a": spanner.param_types.STRING}
    rows = spanner_query(sql, params=params, param_types=ptypes)
    return int(rows[0][0]) if rows else 0


# ----------------------------
# Embedding
# ----------------------------
def _prep_text(text: str) -> str:
    s = (text or "").strip()
    return s[:MAX_EMBED_CHARS]


def embed_one_text(text: str) -> List[float]:
    client = genai_client()
    text = _prep_text(text)
    if not text:
        raise ValueError("empty chunk")

    if genai_types:
        config = genai_types.EmbedContentConfig(
            output_dimensionality=EMBED_DIM,
            task_type="RETRIEVAL_DOCUMENT",
            auto_truncate=True,
        )
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=config,
        )
    else:
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config={
                "outputDimensionality": EMBED_DIM,
                "taskType": "RETRIEVAL_DOCUMENT",
                "autoTruncate": True,
            },
        )

    vec = [float(x) for x in resp.embeddings[0].values]
    if len(vec) != EMBED_DIM:
        raise ValueError("embedding dimension mismatch")
    return vec


# ----------------------------
# Writes
# ----------------------------
def write_embeddings_batch(rows):
    if not rows:
        return
    with spanner_db().batch() as batch:
        batch.update(
            table="FilingChunk",
            columns=("Ticker", "AccessionNo", "ChunkId", "Embedding"),
            values=rows,
        )


def mark_filing_status(ticker, accession, status):
    with spanner_db().batch() as batch:
        batch.update(
            table="Filing",
            columns=("Ticker", "AccessionNo", "Status", "UpdatedAt"),
            values=[(ticker, accession, status, spanner.COMMIT_TIMESTAMP)],
        )


# ----------------------------
# Job
# ----------------------------
def run_job(dry_run: bool, rid: str, req_ticker: str | None):
    t0 = time.time()

    chunks = fetch_unembedded_chunks(MAX_CHUNKS_PER_RUN, req_ticker)
    log.info("[%s] picked %d chunks (ticker=%s)", rid, len(chunks), req_ticker)

    embedded = 0
    errors = []
    touched = set()
    errors_by_filing = defaultdict(int)
    pending = []

    for c in chunks:
        t, a, cid, txt = c["ticker"], c["accession"], c["chunk_id"], c["text"]
        touched.add((t, a))

        try:
            if not dry_run:
                vec = embed_one_text(txt)
                pending.append((t, a, cid, vec))
                if len(pending) >= SPANNER_WRITE_BATCH:
                    write_embeddings_batch(pending)
                    pending.clear()
            embedded += 1
        except Exception as e:
            log.exception("[%s] embed failed %s %s %s", rid, t, a, cid)
            errors.append({"ticker": t, "accession": a, "chunk_id": cid, "error": str(e)})
            errors_by_filing[(t, a)] += 1

        if SLEEP_BETWEEN_CALLS_SEC:
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    if not dry_run and pending:
        write_embeddings_batch(pending)

    if not dry_run and UPDATE_FILING_STATUS:
        for (t, a) in touched:
            if errors_by_filing.get((t, a)):
                mark_filing_status(t, a, "ERROR")
            elif remaining_unembedded_count_for_filing(t, a) == 0:
                mark_filing_status(t, a, "EMBEDDED")

    return {
        "request_id": rid,
        "utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "picked_chunks": len(chunks),
        "embedded_chunks": embedded,
        "touched_filings": len(touched),
        "errors": errors,
        "seconds": round(time.time() - t0, 3),
    }


# ----------------------------
# Flask
# ----------------------------
@app.route("/run", methods=["POST"])
def run():
    rid = uuid.uuid4().hex[:8]
    dry_run = request.args.get("dry_run", "false").lower() == "true"

    if CRON_SECRET:
        if request.headers.get("X-Cron-Secret") != CRON_SECRET:
            return jsonify({"error": "forbidden"}), 403

    body = request.get_json(silent=True) or {}
    req_ticker = (body.get("test_ticker") or "").strip().upper() or None

    out = run_job(dry_run=dry_run, rid=rid, req_ticker=req_ticker)
    return jsonify(out), 200


@app.route("/", methods=["GET"])
def root():
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=True)
