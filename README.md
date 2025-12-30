# Company Filing Pipeline

## Services

### ingest-sec
- Checks SEC for new filings
- Downloads filings to GCS
- Updates FilingIngestState in Spanner

### chunk-to-spanner
- Reads filings from GCS
- Extracts + chunks text
- Writes Filing + FilingChunk rows into Spanner

## Shared
Reusable utilities for SEC, GCS, Spanner, and text processing.
