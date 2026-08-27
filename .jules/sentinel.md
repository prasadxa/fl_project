## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-03-23 - [API Route Vulnerability Pattern]
**Vulnerability:** Unauthenticated access to sensitive admin API routes.
**Learning:** Admin endpoints were left completely unauthenticated, exposing aggregate statistics, overriding user feedback, and entire datasets to the public web via `/api/admin/*`.
**Prevention:** Always implement basic authentication or token validation via FastAPI dependencies (e.g., `Depends(verify_admin)`) on sensitive routes handling internal data, logs, or exports.
