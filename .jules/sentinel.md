## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-24 - [Unauthenticated Admin Endpoints]
**Vulnerability:** The backend endpoints under `/api/admin/*` were completely unprotected, allowing unauthorized access to aggregate stats, feedback logs, and prediction sessions.
**Learning:** Hardcoded endpoints lack default security middleware if not explicitly protected using FastAPI dependencies.
**Prevention:** Always use `Depends(security)` (like `HTTPBasic`) coupled with secure credential comparison (`secrets.compare_digest()`) and fail-secure logic (raising HTTP 500 if credentials are unconfigured) on sensitive administrative routes.
