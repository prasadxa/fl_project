## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-09-25 - [Unauthenticated Admin Endpoints]
**Vulnerability:** Admin endpoints for stats, feedback, export, and sessions lacked authentication, allowing any user to view sensitive data.
**Learning:** In FastAPI, routes meant for admin use must explicitly require a security dependency, as path matching does not inherently enforce access control.
**Prevention:** Always add a security dependency like `Depends(verify_admin)` using `HTTPBasic` or OAuth2 for any endpoint dealing with admin actions or sensitive reports.
