import os
import time
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify
from google.cloud import spanner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ----------------------------
# Env
# ----------------------------
SPANNER_PROJECT = os.environ["SPANNER_PROJECT"]
SPANNER_INSTANCE = os.environ["SPANNER_INSTANCE"]
SPANNER_DATABASE = os.environ["SPANNER_DATABASE"]

CRON_SECRET = os.environ.get("CRON_SECRET", "")

# Retention (days). Default = 180 (~6 months)
RETENTION_DAYS_DEFAULT = int(os.environ.get("RETENTION_DAYS", "180"))

# Safety caps per run (so one call can't delete too much)
MAX_ROWS_PER_RUN = int(os.environ.get("MAX_ROWS_PER_RUN", "20000"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "500"))  # rows per transaction

app = Flask(__name__)
_spanner_db = None


# ----------------------------
# Spanner client helpers
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


# ----------------------------
# Core cleanup logic
# ----------------------------
def _cutoff_ts(retention_days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=retention_days)


def count_eligible_chunks(cutoff: datetime, test_ticker: str) -> int:
    where_ticker = ""
    params = {"cutoff": cutoff}
    ptypes = {"cutoff": spanner.param_types.TIMESTAMP}

    if test_ticker:
        where_ticker = " AND f.Ticker = @t"
        params["t"] = test_ticker
        ptypes["t"] = spanner.param_types.STRING

    sql = f"""
      SELECT COUNT(1)
      FROM FilingChunk fc
      JOIN Filing f
        ON f.Ticker = fc.Ticker AND f.AccessionNo = fc.AccessionNo
      WHERE f.FiledAt IS NOT NULL
        AND f.FiledAt < @cutoff
        {where_ticker}
    """
    rows = spanner_query(sql, params=params, param_types=ptypes)
    return int(rows[0][0]) if rows else 0


def delete_chunk_batch(cutoff: datetime, test_ticker: str, rid: str) -> int:
    """
    Delete up to BATCH_SIZE FilingChunk rows that belong to filings older than cutoff.
    Returns number of chunks deleted in this transaction.
    """
    db = spanner_db()

    def _txn(txn):
        where_ticker = ""
        params = {"cutoff": cutoff, "lim": BATCH_SIZE}
        ptypes = {"cutoff": spanner.param_types.TIMESTAMP, "lim": spanner.param_types.INT64}

        if test_ticker:
            where_ticker = " AND f.Ticker = @t"
            params["t"] = test_ticker
            ptypes["t"] = spanner.param_types.STRING

        # Select keys to delete (bounded)
        sel = f"""
          SELECT fc.Ticker, fc.AccessionNo, fc.ChunkId
          FROM FilingChunk fc
          JOIN Filing f
            ON f.Ticker = fc.Ticker AND f.AccessionNo = fc.AccessionNo
          WHERE f.FiledAt IS NOT NULL
            AND f.FiledAt < @cutoff
            {where_ticker}
          ORDER BY fc.Ticker, fc.AccessionNo, fc.ChunkId
          LIMIT @lim
        """
        rows = list(txn.execute_sql(sel, params=params, param_types=ptypes))
        if not rows:
            return 0

        # Delete by primary keys
        keys = [(r[0], r[1], int(r[2])) for r in rows]
        txn.delete("FilingChunk", spanner.KeySet(keys=keys))
        return len(keys)

    deleted = db.run_in_transaction(_txn)
    log.info("[%s] deleted %d chunk rows (batch)", rid, deleted)
    return deleted


def run_cleanup(dry_run: bool, rid: str, retention_days: int, test_ticker: str) -> Dict[str, Any]:
    t0 = time.time()
    cutoff = _cutoff_ts(retention_days)

    eligible = count_eligible_chunks(cutoff, test_ticker)
    log.info("[%s] eligible chunks=%d (retention_days=%d, test_ticker=%s)", rid, eligible, retention_days, test_ticker or "ALL")

    if dry_run:
        return {
            "request_id": rid,
            "utc": datetime.now(timezone.utc).isoformat(),
            "dry_run": True,
            "retention_days": retention_days,
            "cutoff_utc": cutoff.isoformat(),
            "test_ticker": test_ticker or "",
            "eligible_chunks": eligible,
            "deleted_chunks": 0,
            "seconds": round(time.time() - t0, 3),
        }

    deleted_total = 0
    loops = 0

    # Keep deleting in small transactions until:
    # - nothing left in scope, or
    # - we hit MAX_ROWS_PER_RUN safety cap
    while deleted_total < MAX_ROWS_PER_RUN:
        loops += 1
        deleted = delete_chunk_batch(cutoff, test_ticker, rid)
        if deleted == 0:
            break
        deleted_total += deleted

    return {
        "request_id": rid,
        "utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": False,
        "retention_days": retention_days,
        "cutoff_utc": cutoff.isoformat(),
        "test_ticker": test_ticker or "",
        "eligible_chunks": eligible,
        "deleted_chunks": deleted_total,
        "batches": loops,
        "max_rows_per_run": MAX_ROWS_PER_RUN,
        "batch_size": BATCH_SIZE,
        "seconds": round(time.time() - t0, 3),
    }


# ----------------------------
# Flask endpoints
# ----------------------------
@app.route("/run", methods=["POST"])
def run():
    rid = uuid.uuid4().hex[:8]

    # Security: shared secret header (same pattern as your other services)
    if CRON_SECRET:
        hdr = request.headers.get("X-Cron-Secret", "")
        if hdr != CRON_SECRET:
            return jsonify({"error": "forbidden"}), 403

    # Input
    dry_run = request.args.get("dry_run", "false").lower() == "true"
    body = request.get_json(silent=True) or {}

    retention_days = int(body.get("retention_days", RETENTION_DAYS_DEFAULT))
    if retention_days < 1:
        return jsonify({"error": "retention_days must be >= 1"}), 400

    test_ticker = (body.get("test_ticker") or "").strip().upper()

    log.info("[%s] /run cleanup (dry_run=%s retention_days=%d test_ticker=%s)", rid, dry_run, retention_days, test_ticker or "ALL")
    out = run_cleanup(dry_run=dry_run, rid=rid, retention_days=retention_days, test_ticker=test_ticker)
    return jsonify(out), 200


@app.route("/", methods=["GET"])
def root():
    return "ok", 200
