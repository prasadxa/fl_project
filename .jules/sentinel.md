## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-03-22 - [Missing Authentication on Admin Endpoints]
**Vulnerability:** The `/api/admin/*` endpoints were publicly accessible without any authentication, leaking sensitive feedback, stats, and prediction sessions.
**Learning:** Critical internal APIs must enforce authentication by default to prevent unauthorized access.
**Prevention:** Always use standard auth dependencies like `HTTPBasic` to secure sensitive administrative endpoints.
