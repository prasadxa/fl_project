## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2026-09-27 - [Missing Authentication on Admin Endpoints]
**Vulnerability:** Admin endpoints for stats, feedback, and exports lack authentication, exposing sensitive data.
**Learning:** In FastAPI, relying on hidden frontend routing is insufficient; endpoints must explicitly implement auth dependencies to prevent direct API access.
**Prevention:** Always apply 'Depends(verify_admin)' or similar auth logic directly to sensitive route handlers.
