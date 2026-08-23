## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-03-24 - [Missing Authentication on Admin API Endpoints]
**Vulnerability:** Admin API endpoints (`/api/admin/*`) were fully unauthenticated, allowing any user to download raw medical feedback, user sessions, and comprehensive excel reports without supplying credentials.
**Learning:** Even if frontend endpoints are hidden or require client-side auth, backend API endpoints must enforce authentication internally (e.g. using HTTPBasic and `secrets.compare_digest`), as malicious users can query them directly.
**Prevention:** Enforce robust backend route guards (e.g., using FastAPI's `Depends`) with constant-time string comparison for credentials rather than standard string equality (`==`) to mitigate timing attacks, and implement a fallback response of 500 when server environment variables are missing.
