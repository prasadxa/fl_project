## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-24 - Missing Authentication on Admin API
**Vulnerability:** The /api/admin/* endpoints were unauthenticated, allowing public users to access sensitive logs and system statistics.
**Learning:** Admin routes must be strictly segregated and protected using HTTPBasic or token authentication. Furthermore, authentication credentials should fail-secure (returning HTTP 500) if environment variables are unconfigured, rather than using weak default fallbacks.
**Prevention:** Always implement `Depends(verify_admin)` on any newly added /api/admin/* routes and ensure fail-secure credential loading.
