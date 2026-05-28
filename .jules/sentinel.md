## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-18 - [Fix Unauthenticated Admin Endpoints]
**Vulnerability:** The /api/admin/* endpoints were completely unauthenticated, exposing aggregate feedback statistics, session logs, and full DB exports (CSV/Excel) to anyone who discovered the URLs.
**Learning:** Never assume an endpoint's obscurity protects it. Adding a "secret" frontend route without a backend authorization check means all the data is effectively public.
**Prevention:** Always secure admin or sensitive routes with explicit authorization mechanisms, like FastAPI's HTTPBasic or OAuth2 dependencies, and fail securely (e.g., HTTP 500) if required secrets are unconfigured.
