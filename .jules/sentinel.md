## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-24 - [Missing Admin Authentication]
**Vulnerability:** The /api/admin endpoints are unauthenticated, allowing any user to view sensitive feedback data and prediction sessions, and download Excel/CSV reports.
**Learning:** FastAPI endpoints do not inherit security from their tags or prefixes automatically. An explicit dependency must be provided to enforce authentication.
**Prevention:** Always add a security dependency like Depends(verify_admin) to administrative or sensitive route definitions.
