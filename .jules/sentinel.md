## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2026-07-30 - Unprotected Admin Endpoints Exposed
**Vulnerability:** The FastAPI backend admin endpoints (`/api/admin/stats`, `/api/admin/feedback`, etc.) lacked authentication, exposing sensitive usage statistics, telemetry, and direct access to feedback entries.
**Learning:** Admin endpoints were left completely unprotected while UI functionality was assumed as the sole gating mechanism. Security layers should exist on the API boundaries independently of the UI.
**Prevention:** Implement endpoint-level access controls natively in backend routers (e.g., via FastAPI's `Depends(HTTPBasic)` reading credentials from secure environment variables) rather than relying on frontend routing obscurity.
