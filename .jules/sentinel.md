## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-24 - [Missing Authentication on Admin Endpoints]
**Vulnerability:** Admin API endpoints (`/api/admin/*`) lacked any authentication, exposing sensitive aggregate statistics, feedback logs, and prediction sessions to public access.
**Learning:** Even if the frontend UI hides links, direct API requests to sensitive endpoints must be protected using backend validation/authentication.
**Prevention:** Always secure administrative API endpoints with authentication mechanisms (e.g., HTTP Basic Auth) and fail securely (e.g., 500 error) if required credentials are not configured in the environment.
