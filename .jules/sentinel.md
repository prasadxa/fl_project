## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-03-24 - [Missing Authentication on Admin Endpoints]
**Vulnerability:** Admin endpoints for stats, feedback, and exports lacked authentication allowing any unauthenticated user to access sensitive metrics and logs.
**Learning:** Security by obscurity is insufficient. All admin or sensitive routes must explicitly require an authentication dependency that strictly enforces credentials configured via environment variables.
**Prevention:** Always secure admin routes with authentication middleware or dependencies like HTTPBasic and verify credentials securely, ensuring no sensitive data routes are left open.
