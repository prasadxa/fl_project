## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-24 - Missing Authentication on Admin Endpoints
**Vulnerability:** Admin endpoints (`/api/admin/*`) were accessible without any authentication, exposing sensitive aggregate statistics, feedback logs, and prediction sessions.
**Learning:** When building internal dashboards, endpoints must be secured. Relying on obscurity is not sufficient.
**Prevention:** Implement `HTTPBasic` authentication using FastAPI's dependency injection. Ensure to use `secrets.compare_digest` to prevent timing attacks, and `auto_error=False` to handle missing environment variables securely by returning a 500 error instead of failing open or throwing an unhandled exception.
