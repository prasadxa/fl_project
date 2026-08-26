## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-27 - [Missing Authentication on Admin Endpoints]
**Vulnerability:** The `/api/admin/*` endpoints were accessible without any authentication, exposing sensitive feedback data, sessions, and administrative statistics.
**Learning:** In a full-stack application, ensuring all privileged backend endpoints enforce authentication is critical. A lack of authentication controls leaves the system open to unauthorized data access and potential abuse by unauthenticated actors.
**Prevention:** Always implement an authentication mechanism (e.g., HTTP Basic Authentication or token-based auth) for administrative routes using `Depends` in FastAPI, and securely handle credentials using robust comparison methods like `secrets.compare_digest` to prevent timing attacks.
