## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2026-10-14 - [Missing Authentication on Admin API]
**Vulnerability:** Admin endpoints for stats, feedback, sessions, and exports were fully unauthenticated and exposed publicly.
**Learning:** Always explicitly protect administrative or sensitive data endpoints using `Depends` in FastAPI, even if the frontend requires routing or navigation knowledge to access them.
**Prevention:** Use `fastapi.security.HTTPBasic` (or stronger auth) and apply it to all sensitive routes using a dependency injection.
