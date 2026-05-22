## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-03-22 - [Unauthenticated Admin Endpoints]
**Vulnerability:** Admin endpoints under /api/admin/* lacked authentication, allowing unauthorized users to access and export sensitive system statistics, logs, and diagnostic data.
**Learning:** Basic Auth combined with environment variables provides a straightforward defense. It is vital to use fail-secure architecture (rejecting all requests if credentials are not configured) and to prevent timing attacks by using `secrets.compare_digest()`.
**Prevention:** Apply the `Depends(verify_admin)` dependency to all sensitive endpoints, and ensure `verify_admin` validates that credentials are both present in the environment and match securely.
