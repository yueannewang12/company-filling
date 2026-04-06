import os
import re
import sys
import time
import uuid
import json
import logging
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from flask import Flask, request, jsonify
from google.cloud import storage
from google.cloud import spanner

# allow "from shared import ..." later if you want
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ----------------------------
# Env
# ----------------------------
GCS_BUCKET = os.environ["GCS_BUCKET"]

SPANNER_PROJECT = os.environ["SPANNER_PROJECT"]
SPANNER_INSTANCE = os.environ["SPANNER_INSTANCE"]
SPANNER_DATABASE = os.environ["SPANNER_DATABASE"]

CRON_SECRET = os.environ.get("CRON_SECRET", "")

TEST_TICKER = os.environ.get("TEST_TICKER", "").strip().upper()

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "8000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "400"))

LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "7"))
MAX_FILINGS_PER_COMPANY = int(os.environ.get("MAX_FILINGS_PER_COMPANY", "10"))

FORCE_REPROCESS = os.environ.get("FORCE_REPROCESS", "false").lower() == "true"
RETRY_ERRORS = os.environ.get("RETRY_ERRORS", "false").lower() == "true"

DONE_STATUSES = {"CHUNKED", "EMBEDDED"}

app = Flask(__name__)

_storage_client = None
_spanner_db = None

# ----------------------------
# Impact classification
# ----------------------------
TIER_1_FORMS = {"8-K", "10-Q", "10-K", "6-K", "20-F"}
TIER_2_PREFIXES = ("S-1", "S-3", "F-1", "F-3", "424B", "DEF 14A", "SC 13D", "SC 13G")


def impact_tier(form: str) -> int:
    f = (form or "").strip().upper()
    if f in TIER_1_FORMS:
        return 1
    if any(f.startswith(p) for p in TIER_2_PREFIXES):
        return 2
    return 3


ITEM_PATTERNS = [
    ("earnings", re.compile(r"\bITEM\s+2\.02\b", re.IGNORECASE)),
    ("bankruptcy", re.compile(r"\bITEM\s+1\.03\b|\bBANKRUPTCY\b", re.IGNORECASE)),
    ("financing", re.compile(r"\bITEM\s+2\.03\b|\bCREDIT (AGREEMENT|FACILITY)\b|\bDEBT\b", re.IGNORECASE)),
    ("material_agreement", re.compile(r"\bITEM\s+1\.01\b|\bMATERIAL DEFINITIVE AGREEMENT\b", re.IGNORECASE)),
    ("exec_change", re.compile(r"\bITEM\s+5\.02\b|\bCHIEF (EXECUTIVE|FINANCIAL) OFFICER\b", re.IGNORECASE)),
    ("mna", re.compile(r"\bACQUISITION\b|\bMERGER\b", re.IGNORECASE)),
]


def impact_signals_for_text(form: str, raw_text: str) -> List[str]:
    f = (form or "").strip().upper()
    if f != "8-K":
        return []
    s = raw_text or ""
    out = []
    seen = set()
    for name, pat in ITEM_PATTERNS:
        if pat.search(s) and name not in seen:
            out.append(name)
            seen.add(name)
    return out


# ----------------------------
# Clients
# ----------------------------
def storage_client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client


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
# Helpers
# ----------------------------
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_gcs_filing_path(blob_name: str) -> Optional[Tuple[str, str, str]]:
    parts = blob_name.split("/")
    if len(parts) != 4:
        return None
    ticker, run_dt, accession, filename = parts
    if not DATE_RE.match(run_dt):
        return None
    if filename != f"{accession}.txt":
        return None
    if "-" not in accession:
        return None
    return ticker.upper(), run_dt, accession


def list_recent_filing_blobs_for_ticker(ticker: str, lookback_days: int, rid: str) -> List[storage.Blob]:
    bucket = storage_client().bucket(GCS_BUCKET)
    prefix = f"{ticker}/"
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=lookback_days))

    blobs: List[storage.Blob] = []
    t0 = time.time()
    for blob in bucket.list_blobs(prefix=prefix):
        if not blob.name.endswith(".txt"):
            continue
        parsed = parse_gcs_filing_path(blob.name)
        if not parsed:
            continue
        _, run_dt, _ = parsed
        try:
            d = date.fromisoformat(run_dt)
        except Exception:
            continue
        if d >= cutoff:
            blobs.append(blob)

    blobs.sort(key=lambda b: b.name, reverse=True)
    log.info("[%s] GCS scan %s: found %d candidate .txt blobs in %.2fs",
             rid, ticker, len(blobs), time.time() - t0)
    return blobs


def gcs_read_bytes(blob_name: str, rid: str) -> bytes:
    bucket = storage_client().bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)
    t0 = time.time()
    data = blob.download_as_bytes()
    log.info("[%s] GCS read gs://%s/%s (%d bytes) in %.2fs",
             rid, GCS_BUCKET, blob_name, len(data), time.time() - t0)
    return data


def get_company_list(rid: str, req_ticker: str = "") -> List[Tuple[str, str]]:
    sql = "SELECT Ticker, CIK FROM Company WHERE IsActive = TRUE"
    rows = spanner_query(sql)
    companies = [(r[0], r[1]) for r in rows]
    # Optional request-scoped ticker filter (workflow/body override)
    effective_ticker = (req_ticker or "").strip().upper() or TEST_TICKER
    if effective_ticker:
        companies = [c for c in companies if c[0].upper() == effective_ticker]
        if req_ticker:
            log.info("[%s] Request filter: selecting only ticker=%s", rid, effective_ticker)
        else:
            log.info("[%s] ENV TEST mode: selecting only TEST_TICKER=%s", rid, effective_ticker)

    return companies


def get_filing_status(ticker: str, accession: str) -> Optional[str]:
    sql = """
      SELECT Status
      FROM Filing
      WHERE Ticker = @t AND AccessionNo = @a
      LIMIT 1
    """
    params = {"t": ticker, "a": accession}
    ptypes = {"t": spanner.param_types.STRING, "a": spanner.param_types.STRING}
    rows = spanner_query(sql, params=params, param_types=ptypes)
    if not rows:
        return None
    return rows[0][0]


def get_filing_impact_tier(ticker: str, accession: str) -> Optional[int]:
    sql = """
      SELECT ImpactTier
      FROM Filing
      WHERE Ticker=@t AND AccessionNo=@a
      LIMIT 1
    """
    params = {"t": ticker, "a": accession}
    ptypes = {"t": spanner.param_types.STRING, "a": spanner.param_types.STRING}
    rows = spanner_query(sql, params=params, param_types=ptypes)
    if not rows:
        return None
    v = rows[0][0]
    return int(v) if v is not None else None


# ----------------------------
# Metadata + text extraction
# ----------------------------
def extract_header_value(text: str, key: str) -> Optional[str]:
    m = re.search(rf"^{re.escape(key)}\s*:\s*(.+)$", text, flags=re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip()


def parse_submission_metadata(raw_text: str) -> Dict:
    form = extract_header_value(raw_text, "CONFORMED SUBMISSION TYPE")
    filed = extract_header_value(raw_text, "FILED AS OF DATE")
    period = extract_header_value(raw_text, "CONFORMED PERIOD OF REPORT")

    filed_at = None
    if filed and re.match(r"^\d{8}$", filed):
        y, mo, d = int(filed[0:4]), int(filed[4:6]), int(filed[6:8])
        filed_at = datetime(y, mo, d, 0, 0, 0, tzinfo=timezone.utc)

    period_end = None
    if period and re.match(r"^\d{8}$", period):
        y, mo, d = int(period[0:4]), int(period[4:6]), int(period[6:8])
        period_end = date(y, mo, d)

    return {"FormType": form, "FiledAt": filed_at, "PeriodEnd": period_end}


TAG_RE = re.compile(r"<[^>]+>")


def html_to_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = TAG_RE.sub(" ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_primary_text_from_submission(raw_text: str) -> str:
    text_blocks = re.findall(r"<TEXT>(.*?)</TEXT>", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if text_blocks:
        biggest = max(text_blocks, key=lambda x: len(x))
        return html_to_text(biggest)
    return html_to_text(raw_text)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[Tuple[int, int, str]]:
    if not text:
        return []
    chunks: List[Tuple[int, int, str]] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


# ----------------------------
# Spanner write (idempotent)
# ----------------------------
def upsert_filing_and_chunks(
    ticker: str,
    cik: str,
    accession: str,
    form_type: Optional[str],
    filed_at: Optional[datetime],
    period_end: Optional[date],
    raw_doc_gcs_path: str,
    chunks: List[Tuple[int, int, str]],
    impact_tier_val: int,
    impact_signals: List[str],
    rid: str,
    dry_run: bool,
):
    if dry_run:
        log.info("[%s] DRY RUN: would write Filing + %d chunks for %s %s",
                 rid, len(chunks), ticker, accession)
        return

    db = spanner_db()

    def _txn_fn(txn):
        txn.execute_update(
            "DELETE FROM FilingChunk WHERE Ticker=@t AND AccessionNo=@a",
            params={"t": ticker, "a": accession},
            param_types={"t": spanner.param_types.STRING, "a": spanner.param_types.STRING},
        )

        raw_json = {"source": "gcs-submission-txt", "request_id": rid}
        raw_json_str = json.dumps(raw_json)

        txn.insert_or_update(
            table="Filing",
            columns=(
                "Ticker",
                "AccessionNo",
                "FormType",
                "FiledAt",
                "PeriodEnd",
                "RawDocGcsPath",
                "Status",
                "ImpactTier",
                "ImpactSignals",
                "RawJson",
                "DetectedAt",
                "UpdatedAt",
            ),
            values=[
                (
                    ticker,
                    accession,
                    (form_type or "UNKNOWN")[:12],
                    filed_at,
                    period_end,
                    raw_doc_gcs_path,
                    "CHUNKED",
                    impact_tier_val,
                    json.dumps(impact_signals),
                    raw_json_str,
                    spanner.COMMIT_TIMESTAMP,
                    spanner.COMMIT_TIMESTAMP,
                )
            ],
        )

        for idx, (start_off, end_off, chunk_txt) in enumerate(chunks, start=1):
            txn.insert(
                table="FilingChunk",
                columns=(
                    "Ticker",
                    "AccessionNo",
                    "ChunkId",
                    "Section",
                    "ChunkText",
                    "StartOffset",
                    "EndOffset",
                    "CreatedAt",
                ),
                values=[
                    (
                        ticker,
                        accession,
                        idx,
                        None,
                        chunk_txt,
                        start_off,
                        end_off,
                        spanner.COMMIT_TIMESTAMP,
                    )
                ],
            )

    db.run_in_transaction(_txn_fn)


def mark_filing_error(
    ticker: str,
    accession: str,
    raw_doc_gcs_path: str,
    error_msg: str,
    rid: str,
    dry_run: bool,
):
    if dry_run:
        log.info("[%s] DRY RUN: would mark ERROR for %s %s: %s", rid, ticker, accession, error_msg)
        return

    db = spanner_db()

    def _txn_fn(txn):
        raw_json = {
            "source": "chunk-to-spanner",
            "request_id": rid,
            "error": error_msg[:4000],
        }
        txn.insert_or_update(
            table="Filing",
            columns=(
                "Ticker",
                "AccessionNo",
                "FormType",
                "RawDocGcsPath",
                "Status",
                "RawJson",
                "DetectedAt",
                "UpdatedAt",
            ),
            values=[
                (
                    ticker,
                    accession,
                    "UNKNOWN",
                    raw_doc_gcs_path,
                    "ERROR",
                    json.dumps(raw_json),
                    spanner.COMMIT_TIMESTAMP,
                    spanner.COMMIT_TIMESTAMP,
                )
            ],
        )

    db.run_in_transaction(_txn_fn)


# ----------------------------
# Main job
# ----------------------------
def run_job(dry_run: bool, rid: str, req_ticker: str = "") -> Dict:
    t0 = time.time()

    companies = get_company_list(rid, req_ticker=req_ticker)
    log.info("[%s] Will process %d companies", rid, len(companies))

    processed_filings = 0
    skipped_done = 0
    skipped_error = 0
    errors = []

    for ticker, cik in companies:
        log.info("[%s] ---- Company start: %s (CIK=%s) ----", rid, ticker, cik)
        try:
            blobs = list_recent_filing_blobs_for_ticker(ticker, LOOKBACK_DAYS, rid)

            count_considered = 0
            for blob in blobs:
                parsed = parse_gcs_filing_path(blob.name)
                if not parsed:
                    continue
                _, _, accession = parsed

                if count_considered >= MAX_FILINGS_PER_COMPANY:
                    break
                count_considered += 1

                status = get_filing_status(ticker, accession)
                if not FORCE_REPROCESS:
                    if status in DONE_STATUSES:
                        skipped_done += 1
                        continue
                    if status == "ERROR" and not RETRY_ERRORS:
                        skipped_error += 1
                        continue

                try:
                    raw_bytes = gcs_read_bytes(blob.name, rid)
                    raw_text = raw_bytes.decode("utf-8", errors="ignore")

                    meta = parse_submission_metadata(raw_text)
                    form_type = meta.get("FormType") or "UNKNOWN"
                    filed_at = meta.get("FiledAt")
                    period_end = meta.get("PeriodEnd")

                    # tier: preserve if already exists, otherwise compute
                    existing_tier = get_filing_impact_tier(ticker, accession)
                    tier = existing_tier if existing_tier is not None else impact_tier(form_type)

                    # signals: compute here while we already have raw_text
                    signals = impact_signals_for_text(form_type, raw_text)

                    cleaned = extract_primary_text_from_submission(raw_text)
                    chunks = chunk_text(cleaned, CHUNK_SIZE, CHUNK_OVERLAP)

                    log.info("[%s] %s %s: extracted %d chars, %d chunks (tier=%s signals=%s)",
                             rid, ticker, accession, len(cleaned), len(chunks), tier, signals)

                    upsert_filing_and_chunks(
                        ticker=ticker,
                        cik=cik,
                        accession=accession,
                        form_type=form_type,
                        filed_at=filed_at,
                        period_end=period_end,
                        raw_doc_gcs_path=blob.name,
                        chunks=chunks,
                        impact_tier_val=int(tier),
                        impact_signals=signals,
                        rid=rid,
                        dry_run=dry_run,
                    )
                    processed_filings += 1

                except Exception as fe:
                    msg = str(fe)
                    log.exception("[%s] Filing failed: %s %s", rid, ticker, accession)
                    mark_filing_error(
                        ticker=ticker,
                        accession=accession,
                        raw_doc_gcs_path=blob.name,
                        error_msg=msg,
                        rid=rid,
                        dry_run=dry_run,
                    )
                    errors.append({"ticker": ticker, "accession": accession, "error": msg})
                    continue

        except Exception as e:
            log.exception("[%s] Company failed: %s", rid, ticker)
            errors.append({"ticker": ticker, "cik": cik, "error": str(e)})

    result = {
        "request_id": rid,
        "utc": datetime.utcnow().isoformat() + "+00:00",
        "dry_run": dry_run,
        "companies_targeted": len(companies),
        "processed_filings": processed_filings,
        "skipped_done": skipped_done,
        "skipped_error": skipped_error,
        "errors": errors,
        "seconds": round(time.time() - t0, 3),
    }
    log.info("[%s] JOB DONE: %s", rid, result)
    return result


# ----------------------------
# Flask endpoints
# ----------------------------
@app.route("/run", methods=["POST"])
def run():
    rid = uuid.uuid4().hex[:8]

    body = request.get_json(silent=True) or {}

    # Allow dry_run via query param (?dry_run=true) OR JSON body {"dry_run": true}
    if "dry_run" in request.args:
        dry_run = request.args.get("dry_run", "false").lower() == "true"
    else:
        dry_run = bool(body.get("dry_run", False))

    # Allow request-scoped ticker filter via JSON body {"test_ticker": "AMD"}
    req_ticker = (body.get("test_ticker") or body.get("ticker") or "").strip().upper()

    if CRON_SECRET:
        hdr = request.headers.get("X-Cron-Secret", "")
        if hdr != CRON_SECRET:
            return jsonify({"error": "forbidden"}), 403

    log.info("[%s] /run received (dry_run=%s, req_ticker=%s)", rid, dry_run, req_ticker or "")
    return jsonify(run_job(dry_run=dry_run, rid=rid, req_ticker=req_ticker)), 200


@app.route("/", methods=["GET"])

def root():
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=True)