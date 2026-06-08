## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2025-06-08 - Added Basic Auth to Admin Endpoints
**Vulnerability:** Five admin endpoints (/api/admin/*) lacked authentication and were accessible to anyone.
**Learning:** High severity risk due to exposed admin functionalities which leaked admin statistics and datasets.
**Prevention:** Always implement basic authentication checks on endpoints that expose administrative/internal controls or datasets and rely on securely parsed headers to do so.
