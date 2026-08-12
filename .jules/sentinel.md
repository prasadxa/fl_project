## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2025-02-12 - Added HTTP Basic Auth for Admin Endpoints
**Vulnerability:** The backend /api/admin/* endpoints lacked authentication, allowing unauthorized users to access aggregate statistics, feedback logs, and export sensitive clinical CSV/Excel data.
**Learning:** Sensitive admin endpoints must be properly secured. The frontend lacked mechanisms to send credentials for these endpoints when requesting downloads via standard `window.open` calls.
**Prevention:** Always secure admin endpoints with authentication. When triggering file downloads from secured endpoints, ensure the requests are authenticated (e.g., using `fetch()` with custom authorization headers and then triggering a blob download).
