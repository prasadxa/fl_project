## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2025-02-23 - Missing Authentication on Admin Endpoints
**Vulnerability:** The /api/admin/* endpoints lacked any authentication, exposing sensitive system statistics, logs, and CSV/Excel exports to unauthenticated users.
**Learning:** Admin routes were grouped but not protected by a shared dependency or middleware, leading to an implicit trust assumption.
**Prevention:** Always require explicit authentication guards (e.g., `Depends(verify_admin)`) on sensitive routes and ensure credential validation relies on secure environment variables, failing closed if missing.
