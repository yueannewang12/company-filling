import os
import json
import time
import logging
import uuid
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import requests
from flask import Flask, request, jsonify

from google.cloud import storage
from google.cloud import spanner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ----------------------------
# Env
# ----------------------------
GCS_BUCKET = os.environ["GCS_BUCKET"]
SEC_USER_AGENT = os.environ["SEC_USER_AGENT"]

SPANNER_PROJECT = os.environ["SPANNER_PROJECT"]
SPANNER_INSTANCE = os.environ["SPANNER_INSTANCE"]
SPANNER_DATABASE = os.environ["SPANNER_DATABASE"]

CRON_SECRET = os.environ.get("CRON_SECRET", "")
TEST_TICKER = os.environ.get("TEST_TICKER", "").strip().upper()
RECENT_FILINGS_PER_COMPANY = int(os.environ.get("RECENT_FILINGS_PER_COMPANY", "3"))

HTTP_TIMEOUT = (5.0, 30.0)

app = Flask(__name__)

_storage_client = None
_spanner_db = None


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
def cik10(cik: str) -> str:
    return str(cik).zfill(10)


def http_get(url: str, rid: str) -> requests.Response:
    headers = {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        # SEC endpoints differ; keep Host consistent with URL to avoid edge issues.
        "Host": "data.sec.gov" if "data.sec.gov" in url else "www.sec.gov",
    }
    log.info("[%s] HTTP GET %s (timeout=%s)", rid, url, HTTP_TIMEOUT)
    resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
    log.info("[%s] HTTP %s %s", rid, resp.status_code, url)
    resp.raise_for_status()
    return resp


def gcs_write_bytes(path: str, content: bytes, rid: str) -> None:
    bucket = storage_client().bucket(GCS_BUCKET)
    blob = bucket.blob(path)
    t0 = time.time()
    blob.upload_from_string(content)
    log.info(
        "[%s] GCS wrote gs://%s/%s (%d bytes) in %.2fs",
        rid,
        GCS_BUCKET,
        path,
        len(content),
        time.time() - t0,
    )


def gcs_exists(path: str) -> bool:
    bucket = storage_client().bucket(GCS_BUCKET)
    return bucket.blob(path).exists()


def accession_to_txt_path(ticker: str, run_date: str, accession: str) -> str:
    return f"{ticker}/{run_date}/{accession}/{accession}.txt"


def build_filing_txt_url(cik: str, accession: str) -> str:
    accession_nodash = accession.replace("-", "")
    cik_int = str(int(cik))  # SEC archive path uses CIK without leading zeros
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{accession}.txt"


# ----------------------------
# Spanner I/O
# ----------------------------
def fetch_company_list(rid: str) -> List[Tuple[str, str]]:
    sql = "SELECT Ticker, CIK FROM Company WHERE IsActive = TRUE"
    log.info("[%s] SPANNER query start: %s", rid, sql)
    t0 = time.time()

    rows = spanner_query(sql)

    log.info("[%s] SPANNER query done: %d rows in %.2fs", rid, len(rows), time.time() - t0)

    companies = [(r[0], r[1]) for r in rows]
    if TEST_TICKER:
        companies = [c for c in companies if c[0].upper() == TEST_TICKER]
        log.info("[%s] TEST mode: selecting only TEST_TICKER=%s", rid, TEST_TICKER)
    return companies


def get_ingest_state(ticker: str) -> Tuple[Optional[date], Optional[str]]:
    sql = """
      SELECT LastFilingDate, LastAccession
      FROM FilingIngestState
      WHERE Ticker = @ticker
    """
    params = {"ticker": ticker}
    param_types = {"ticker": spanner.param_types.STRING}

    rows = spanner_query(sql, params=params, param_types=param_types)
    if not rows:
        return None, None
    return rows[0][0], rows[0][1]


def upsert_ingest_state(ticker: str, cik: str, last_filing_date: date, last_accession: str) -> None:
    db = spanner_db()
    with db.batch() as batch:
        batch.insert_or_update(
            table="FilingIngestState",
            columns=("Ticker", "Cik", "LastFilingDate", "LastAccession", "UpdatedAt"),
            values=[(ticker, cik, last_filing_date, last_accession, spanner.COMMIT_TIMESTAMP)],
        )


# ----------------------------
# SEC parsing
# ----------------------------
def fetch_submissions_json(cik: str, rid: str) -> Dict:
    url = f"https://data.sec.gov/submissions/CIK{cik10(cik)}.json"
    return http_get(url, rid).json()


def parse_recent_filings(submissions: Dict) -> List[Dict]:
    """
    Returns filings in the SEC 'recent' order (most recent first).
    """
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    out = []
    n = min(len(forms), len(filing_dates), len(accessions))
    for i in range(n):
        out.append(
            {
                "form": forms[i],
                "filingDate": filing_dates[i],
                "accessionNumber": accessions[i],
                "primaryDocument": primary_docs[i] if i < len(primary_docs) else "",
            }
        )
    return out


def is_newer_than_state(
    filing_date: date, accession: str, last_date: Optional[date], last_acc: Optional[str]
) -> bool:
    """
    Treat (filing_date, accession) as an ordering key.
    """
    if last_date is None:
        return True
    if filing_date > last_date:
        return True
    if filing_date < last_date:
        return False
    # same date
    if last_acc is None:
        return True
    return accession > last_acc


def pick_new_filings_to_process(
    all_recent: List[Dict],
    last_date: Optional[date],
    last_acc: Optional[str],
    cap: int,
    scan_limit: int = 50,
) -> List[Dict]:
    """
    Look at up to scan_limit recent items, keep those newer than state,
    return up to cap in the original order (most recent first).
    """
    new_filings: List[Dict] = []
    for f in all_recent[:scan_limit]:
        try:
            fd = date.fromisoformat(f["filingDate"])
        except Exception:
            continue
        acc = f.get("accessionNumber", "")
        if not acc:
            continue
        if is_newer_than_state(fd, acc, last_date, last_acc):
            new_filings.append(f)
        if len(new_filings) >= cap:
            break
    return new_filings


# ----------------------------
# Job
# ----------------------------
def ingest_job(dry_run: bool, rid: str) -> Dict:
    t0 = time.time()
    run_date = datetime.utcnow().strftime("%Y-%m-%d")

    companies = fetch_company_list(rid)
    log.info("[%s] Will process %d companies", rid, len(companies))

    processed = 0
    errors = []

    for ticker, cik in companies:
        t_company = time.time()
        log.info("[%s] ---- Company start: %s (CIK=%s) ----", rid, ticker, cik)

        try:
            last_date, last_acc = get_ingest_state(ticker)
            log.info(
                "[%s] State for %s: LastFilingDate=%s LastAccession=%s",
                rid,
                ticker,
                last_date,
                last_acc,
            )

            # Must fetch SEC submissions to know if there are new filings
            submissions = fetch_submissions_json(cik, rid)

            candidates = parse_recent_filings(submissions)
            new_filings = pick_new_filings_to_process(
                all_recent=candidates,
                last_date=last_date,
                last_acc=last_acc,
                cap=RECENT_FILINGS_PER_COMPANY,
                scan_limit=max(RECENT_FILINGS_PER_COMPANY, 10),
            )

            log.info(
                "[%s] %s: %d new filings to download (cap=%d)",
                rid,
                ticker,
                len(new_filings),
                RECENT_FILINGS_PER_COMPANY,
            )

            # NEW LAYER: only write submissions JSON if there are new filings
            if new_filings:
                sub_path = f"{ticker}/{run_date}/sec_submissions_{cik10(cik)}.json"
                if dry_run:
                    log.info(
                        "[%s] DRY RUN: would write submissions JSON -> gs://%s/%s",
                        rid,
                        GCS_BUCKET,
                        sub_path,
                    )
                else:
                    if not gcs_exists(sub_path):
                        gcs_write_bytes(sub_path, json.dumps(submissions).encode("utf-8"), rid)
                    else:
                        log.info("[%s] Submissions JSON already exists, skip: gs://%s/%s", rid, GCS_BUCKET, sub_path)

            if not new_filings:
                processed += 1
                log.info(
                    "[%s] ---- Company done: %s (no new filings) in %.2fs ----",
                    rid,
                    ticker,
                    time.time() - t_company,
                )
                continue

            newest_seen_date = last_date
            newest_seen_acc = last_acc

            for f in new_filings:
                fd = date.fromisoformat(f["filingDate"])
                acc = f["accessionNumber"]

                txt_path = accession_to_txt_path(ticker, run_date, acc)

                if dry_run:
                    log.info(
                        "[%s] DRY RUN: would download %s %s %s -> gs://%s/%s",
                        rid,
                        ticker,
                        f.get("form", ""),
                        acc,
                        GCS_BUCKET,
                        txt_path,
                    )
                else:
                    if gcs_exists(txt_path):
                        log.info("[%s] GCS exists, skip: gs://%s/%s", rid, GCS_BUCKET, txt_path)
                    else:
                        url = build_filing_txt_url(cik, acc)
                        content = http_get(url, rid).content
                        gcs_write_bytes(txt_path, content, rid)

                # Track max (fd, acc)
                if newest_seen_date is None:
                    newest_seen_date, newest_seen_acc = fd, acc
                else:
                    if fd > newest_seen_date:
                        newest_seen_date, newest_seen_acc = fd, acc
                    elif fd == newest_seen_date and (newest_seen_acc is None or acc > newest_seen_acc):
                        newest_seen_date, newest_seen_acc = fd, acc

            if not dry_run and newest_seen_date and newest_seen_acc:
                upsert_ingest_state(ticker, cik10(cik), newest_seen_date, newest_seen_acc)
                log.info(
                    "[%s] Updated state for %s => %s / %s",
                    rid,
                    ticker,
                    newest_seen_date,
                    newest_seen_acc,
                )

            processed += 1
            log.info("[%s] ---- Company done: %s in %.2fs ----", rid, ticker, time.time() - t_company)

        except Exception as e:
            log.exception("[%s] Company failed: %s (CIK=%s)", rid, ticker, cik)
            errors.append({"ticker": ticker, "cik": cik, "error": str(e)})

    result = {
        "request_id": rid,
        "utc": datetime.utcnow().isoformat() + "+00:00",
        "dry_run": dry_run,
        "company_count_targeted": len(companies),
        "processed_companies": processed,
        "errors": errors,
        "seconds": round(time.time() - t0, 3),
    }
    log.info("[%s] JOB DONE: %s", rid, result)
    return result


# ----------------------------
# Routes
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
    return jsonify(ingest_job(dry_run=dry_run, rid=rid)), 200


@app.route("/", methods=["GET"])
def root():
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=True)
