## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-08-29 - [Missing Authentication on Admin Endpoints]
**Vulnerability:** Unauthenticated access to sensitive administrative endpoints (`/api/admin/*`) exposing metrics, feedback data, and user sessions.
**Learning:** Even if administrative features are hidden in the frontend UI, their backing API routes remain exposed and vulnerable to direct enumeration and data scraping if missing explicit authentication guards at the backend level.
**Prevention:** Always implement authentication guards (e.g., HTTP Basic Auth or JWT validation) explicitly on all administrative routes, and avoid relying solely on "security by obscurity" in the client interface.
