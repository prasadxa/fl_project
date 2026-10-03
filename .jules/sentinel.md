## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-03-24 - [Missing Authentication on Admin Endpoints]
**Vulnerability:** Admin dashboard endpoints exposing sensitive clinical usage statistics, session records, and full feedback logs were publicly accessible without authentication.
**Learning:** Legacy endpoints or endpoints built initially for internal dashboarding can inadvertently be exposed without proper gatekeeping, allowing unauthorized access to aggregate operational and clinical metadata.
**Prevention:** Always secure administrative or internal-only API endpoints with authentication (e.g., HTTP Basic Auth for simple admin access) and verify credentials using secure, constant-time comparison methods.
