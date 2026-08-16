## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2025-02-28 - Missing Authentication on Admin Endpoints
**Vulnerability:** The `/api/admin/*` endpoints providing aggregate statistics, feedback logs, and sensitive data exports (CSV/Excel) were completely unprotected, exposing them to unauthorized access.
**Learning:** Relying on security by obscurity or client-side routing logic is insufficient. Sensitive API routes must have explicit server-side authentication barriers. Furthermore, standard file download triggers (`window.open`) in the frontend bypass custom authorization headers, requiring a fetch+blob approach when securing those routes.
**Prevention:** Always secure administrative routes at the framework level (e.g., using FastAPI's `Depends` and `fastapi.security`). When implementing HTTP Basic Auth, explicitly fail securely on missing configuration (rather than falling back to default credentials) and use constant-time string comparisons (`secrets.compare_digest`) for credential verification to mitigate timing attacks.
