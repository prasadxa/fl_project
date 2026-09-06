## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-03-24 - [Missing Auth on Admin Endpoints]
**Vulnerability:** The /api/admin/* endpoints were completely unauthenticated, exposing aggregate stats, session feedback logs, and CSV/Excel exports.
**Learning:** Always implement authentication on privileged data routes before exposing them in an API. Relying on frontend routing obscurity is not security.
**Prevention:** Apply dependency injection guards (e.g. `Depends(verify_admin)`) globally or via routers for administrative functionality.
