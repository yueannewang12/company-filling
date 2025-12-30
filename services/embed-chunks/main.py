import os
import time
import uuid
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from flask import Flask, request, jsonify
from google.cloud import spanner

from google import genai

# Try to import types (newer google-genai versions)
try:
    from google.genai import types as genai_types
except Exception:
    genai_types = None  # We'll fall back to a plain dict config.

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ----------------------------
# Env
# ----------------------------
SPANNER_PROJECT = os.environ["SPANNER_PROJECT"]
SPANNER_INSTANCE = os.environ["SPANNER_INSTANCE"]
SPANNER_DATABASE = os.environ["SPANNER_DATABASE"]

CRON_SECRET = os.environ.get("CRON_SECRET", "")

# Vertex / embeddings
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")

# Spanner DDL vector_length (must match)
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))

# Throughput knobs
SPANNER_WRITE_BATCH = int(os.environ.get("SPANNER_WRITE_BATCH", "200"))  # rows per commit
MAX_CHUNKS_PER_RUN = int(os.environ.get("MAX_CHUNKS_PER_RUN", "256"))    # total chunks per /run
MAX_EMBED_CHARS = int(os.environ.get("MAX_EMBED_CHARS", "12000"))        # truncate text
SLEEP_BETWEEN_CALLS_SEC = float(os.environ.get("SLEEP_BETWEEN_CALLS_SEC", "0"))  # throttle if needed

# Testing / targeting
TEST_TICKER = os.environ.get("TEST_TICKER", "").strip().upper()  # if set, only process that ticker

# Optional: mark Filing.Status="EMBEDDED" once all chunks embedded for that filing
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
        sp = spanner.Client(project=SPANNER_PROJECT)
        inst = sp.instance(SPANNER_INSTANCE)
        _spanner_db = inst.database(SPANNER_DATABASE)
    return _spanner_db


def spanner_query(sql: str, params=None, param_types=None):
    db = spanner_db()
    with db.snapshot() as snap:
        return list(snap.execute_sql(sql, params=params, param_types=param_types))


def genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        # Vertex backend
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
def fetch_unembedded_chunks(limit: int) -> List[Dict[str, Any]]:
    """
    Select chunks that still need embeddings (idempotent).
    """
    where = "Embedding IS NULL"
    params = {"lim": limit}
    ptypes = {"lim": spanner.param_types.INT64}

    if TEST_TICKER:
        where += " AND Ticker = @t"
        params["t"] = TEST_TICKER
        ptypes["t"] = spanner.param_types.STRING

    sql = f"""
      SELECT Ticker, AccessionNo, ChunkId, ChunkText
      FROM FilingChunk
      WHERE {where}
      ORDER BY Ticker, AccessionNo, ChunkId
      LIMIT @lim
    """
    rows = spanner_query(sql, params=params, param_types=ptypes)
    out = []
    for r in rows:
        out.append(
            {
                "ticker": r[0],
                "accession": r[1],
                "chunk_id": int(r[2]),
                "text": r[3] or "",
            }
        )
    return out


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
def _prep_text_for_embedding(text: str) -> str:
    s = (text or "").strip()
    if len(s) > MAX_EMBED_CHARS:
        s = s[:MAX_EMBED_CHARS]
    return s


def embed_one_text(text: str) -> List[float]:
    """
    gemini-embedding-001 is safest one-text-per-request.
    We explicitly request outputDimensionality=EMBED_DIM to match Spanner vector_length.
    """
    client = genai_client()
    text = _prep_text_for_embedding(text)

    if not text:
        raise ValueError("empty text")

    # Prefer typed config if available; otherwise pass a dict.
    if genai_types is not None:
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
        # Fallback: some versions accept config as dict
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config={
                "outputDimensionality": EMBED_DIM,
                "taskType": "RETRIEVAL_DOCUMENT",
                "autoTruncate": True,
            },
        )

    # Response shape: resp.embeddings[0].values
    emb0 = resp.embeddings[0]
    vec = [float(x) for x in emb0.values]

    if len(vec) != EMBED_DIM:
        raise ValueError(
            f"Embedding dim={len(vec)} != EMBED_DIM={EMBED_DIM}. "
            f"Ensure outputDimensionality is set and Spanner vector_length matches."
        )
    return vec


# ----------------------------
# Writes
# ----------------------------
def write_embeddings_batch(rows: List[Tuple[str, str, int, List[float]]]) -> None:
    """
    Update only the Embedding column (plus keys).
    """
    if not rows:
        return
    db = spanner_db()
    with db.batch() as batch:
        batch.update(
            table="FilingChunk",
            columns=("Ticker", "AccessionNo", "ChunkId", "Embedding"),
            values=[(t, a, cid, vec) for (t, a, cid, vec) in rows],
        )


def mark_filing_status(ticker: str, accession: str, status: str) -> None:
    db = spanner_db()
    with db.batch() as batch:
        batch.update(
            table="Filing",
            columns=("Ticker", "AccessionNo", "Status", "UpdatedAt"),
            values=[(ticker, accession, status, spanner.COMMIT_TIMESTAMP)],
        )


# ----------------------------
# Job
# ----------------------------
def run_job(dry_run: bool, rid: str) -> Dict[str, Any]:
    t0 = time.time()

    chunks = fetch_unembedded_chunks(MAX_CHUNKS_PER_RUN)
    log.info(
        "[%s] picked %d chunks (Embedding IS NULL)%s",
        rid,
        len(chunks),
        f" for TEST_TICKER={TEST_TICKER}" if TEST_TICKER else "",
    )

    if not chunks:
        return {
            "request_id": rid,
            "utc": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
            "picked_chunks": 0,
            "embedded_chunks": 0,
            "errors": [],
            "seconds": round(time.time() - t0, 3),
        }

    embedded = 0
    errors: List[Dict[str, Any]] = []

    touched_filings = set()
    errors_by_filing = defaultdict(int)

    pending_updates: List[Tuple[str, str, int, List[float]]] = []

    for item in chunks:
        t = item["ticker"]
        a = item["accession"]
        cid = item["chunk_id"]
        txt = item["text"]

        touched_filings.add((t, a))

        # Empty chunks -> record error, skip
        if not (txt or "").strip():
            errors.append({"ticker": t, "accession": a, "chunk_id": cid, "error": "empty chunk text"})
            errors_by_filing[(t, a)] += 1
            continue

        try:
            if dry_run:
                embedded += 1
                continue

            vec = embed_one_text(txt)
            pending_updates.append((t, a, cid, vec))
            embedded += 1

            # Flush writes in batches
            if len(pending_updates) >= SPANNER_WRITE_BATCH:
                write_embeddings_batch(pending_updates)
                log.info("[%s] wrote %d embeddings", rid, len(pending_updates))
                pending_updates = []

            if SLEEP_BETWEEN_CALLS_SEC > 0:
                time.sleep(SLEEP_BETWEEN_CALLS_SEC)

        except Exception as e:
            log.exception("[%s] embed failed for %s %s chunk=%s", rid, t, a, cid)
            errors.append({"ticker": t, "accession": a, "chunk_id": cid, "error": str(e)})
            errors_by_filing[(t, a)] += 1

    # Flush remaining updates
    if not dry_run and pending_updates:
        write_embeddings_batch(pending_updates)
        log.info("[%s] wrote %d embeddings", rid, len(pending_updates))

    # Optional: mark filings EMBEDDED if fully done, ERROR if any errors for that filing
    if not dry_run and UPDATE_FILING_STATUS and touched_filings:
        for (t, a) in touched_filings:
            try:
                if errors_by_filing.get((t, a), 0) > 0:
                    mark_filing_status(t, a, "ERROR")
                    continue

                remaining = remaining_unembedded_count_for_filing(t, a)
                if remaining == 0:
                    mark_filing_status(t, a, "EMBEDDED")
            except Exception as e:
                log.exception("[%s] status update failed for %s %s", rid, t, a)
                errors.append({"ticker": t, "accession": a, "chunk_id": None, "error": f"status update failed: {e}"})

    return {
        "request_id": rid,
        "utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "picked_chunks": len(chunks),
        "embedded_chunks": embedded if dry_run else embedded,  # embedded count includes successful embeddings
        "touched_filings": len(touched_filings),
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
        hdr = request.headers.get("X-Cron-Secret", "")
        if hdr != CRON_SECRET:
            return jsonify({"error": "forbidden"}), 403

    log.info("[%s] /run received (dry_run=%s)", rid, dry_run)
    out = run_job(dry_run=dry_run, rid=rid)
    return jsonify(out), 200


@app.route("/", methods=["GET"])
def root():
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=True)
