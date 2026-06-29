## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2025-02-28 - Missing Authentication on Admin Endpoints
**Vulnerability:** Admin endpoints in FastAPI backend (`/api/admin/*`) were completely unauthenticated, allowing any user to download full feedback CSVs, Excel reports, and view aggregate stats without authorization.
**Learning:** Even internal or admin-focused API routes must explicitly include authorization dependencies. Utilizing `fastapi.security.HTTPBasic` with environment variable checks allows for simple but robust credential validation.
**Prevention:** Always verify `ADMIN_USER` and `ADMIN_PASS` using `os.getenv()`, explicitly failing secure (HTTP 500) if unconfigured, and use `secrets.compare_digest()` to prevent timing attacks when comparing strings.
