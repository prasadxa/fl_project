## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-04-28 - [Missing Authentication on Admin Endpoints]
**Vulnerability:** Admin API endpoints in FastAPI were missing authentication, exposing sensitive statistics and user feedback data to unauthenticated requests.
**Learning:** All endpoints that expose sensitive data must be explicitly secured, especially when adding new grouped features like admin dashboards.
**Prevention:** Use FastAPI dependencies like `Depends(verify_admin)` uniformly across all sensitive endpoints during implementation.
