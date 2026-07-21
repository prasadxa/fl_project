## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-18 - Missing Authentication on Admin Endpoints
**Vulnerability:** Admin API endpoints (`/api/admin/*`) and the corresponding frontend paths were accessible without any authentication, exposing aggregate statistics, prediction sessions, and full feedback CSV/Excel exports.
**Learning:** The FastAPI backend lacked a security dependency (e.g., `Depends(verify_admin)`) on the admin routes, and the frontend did not prompt for or attach authentication headers. Sensitive operational and potential PII data was exposed to unauthorized users.
**Prevention:** Always require authentication on administrative endpoints and implement role-based access control. Use secure methods to pass credentials (e.g., HTTP Basic Auth over HTTPS or Bearer tokens) and verify them using secure string comparison functions (e.g., `secrets.compare_digest`). Ensure environment variables for sensitive credentials do not fall back to insecure defaults.
