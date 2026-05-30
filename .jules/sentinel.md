## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2025-02-28 - [FastAPI Admin Authentication Bug]
**Vulnerability:** Missing authentication on all /api/admin/* endpoints.
**Learning:** API endpoints for statistics, session logs, and data exports were completely unauthenticated, allowing any client to fetch sensitive data.
**Prevention:** Apply a secure FastAPI HTTPBasic authentication dependency (checking against environment variables with constant-time comparison) to all sensitive routes.
