## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2025-02-28 - Missing Authentication on Admin Endpoints
**Vulnerability:** Admin endpoints were exposed without authentication, allowing unauthorized access to statistics, feedback, and session exports.
**Learning:** In FastAPI, endpoints need explicit security dependencies to prevent unauthorized access.
**Prevention:** Apply  with an appropriate security scheme (like ) to all sensitive routes, ensuring environment variables are securely read and  is used for constant-time comparisons to prevent timing attacks.
## 2025-02-28 - Missing Authentication on Admin Endpoints
**Vulnerability:** Admin endpoints were exposed without authentication, allowing unauthorized access to statistics, feedback, and session exports.
**Learning:** In FastAPI, endpoints need explicit security dependencies to prevent unauthorized access.
**Prevention:** Apply `Depends` with an appropriate security scheme (like `HTTPBasic`) to all sensitive routes, ensuring environment variables are securely read and `secrets.compare_digest` is used for constant-time comparisons to prevent timing attacks.
