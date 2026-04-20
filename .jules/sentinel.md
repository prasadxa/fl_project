## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-20 - Unauthenticated Admin Endpoints
**Vulnerability:** The `/api/admin/*` endpoints were exposed without any authentication, allowing unauthenticated users to access sensitive aggregate statistics, feedback logs, export data, and prediction sessions.
**Learning:** Admin endpoints were left unprotected, potentially due to oversight during development or testing. This is a common pattern where administrative routes are added but security is deferred.
**Prevention:** Always implement authentication and authorization checks (e.g., using `Depends` in FastAPI) when adding any new administrative or sensitive routes, ensuring they are protected by default.
