## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2026-09-15 - Missing Authentication on Admin Endpoints
**Vulnerability:** Admin endpoints under `/api/admin` were completely unprotected, exposing sensitive statistical data, user feedback logs, and application internals to unauthorized users.
**Learning:** Even internal or reporting endpoints must explicitly require authentication, as relying on frontend UI obscurity is inadequate. FastAPI requires explicit dependency injection for route protection.
**Prevention:** Always implement authentication guards (e.g., `fastapi.security.HTTPBasic` or token-based auth) explicitly in the route definition or via a global router dependency for all administrative API namespaces.
