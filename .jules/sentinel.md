## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-03-23 - [Unauthenticated Admin Endpoints]
**Vulnerability:** The `/api/admin/*` endpoints in the backend (e.g. `/api/admin/stats`, `/api/admin/feedback`, export logic) were completely open, allowing unauthenticated access to sensitive aggregate statistics and raw diagnostic feedback without requiring any credentials.
**Learning:** Adding endpoints meant for administrative use under a recognizable path prefix (`/api/admin/`) without explicit authentication middleware or dependencies creates a significant data exposure risk, especially in a system dealing with medical data and clinician feedback.
**Prevention:** Implement explicit authentication gates (e.g. `HTTPBasic` or token-based) on all administrative routes, ensuring proper injection as route dependencies (`Depends(verify_admin)`) so the logic cannot be bypassed. Ensure sensitive parameters like expected credentials are securely retrieved from environment configurations rather than hardcoded.
