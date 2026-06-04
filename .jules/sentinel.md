## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-03-24 - [Add Authentication to Admin Endpoints]
**Vulnerability:** Admin dashboard API routes were missing authentication, exposing sensitive information.
**Learning:** For quick security wins without adding massive dependencies (like OAuth), FastAPI's native `HTTPBasic` along with environment variables (`os.getenv`) and constant-time string comparisons (`secrets.compare_digest`) provide robust protection. Also, migrating unauthenticated frontend file exports (`window.open()`) to authenticated blob downloads (`fetch` -> `res.blob()` -> `URL.createObjectURL()`) ensures tokens are safely transmitted via HTTP headers, not URL parameters.
**Prevention:** Audit all endpoints prefixed with `/admin` to ensure they implement a `Depends(verify_admin)` dependency before merging.
