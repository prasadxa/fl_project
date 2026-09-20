## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-03-24 - [Missing Admin Authentication]
**Vulnerability:** Admin endpoints `/api/admin/*` were completely unauthenticated, exposing patient feedback, AI diagnostic stats, and prediction session logs to anyone.
**Learning:** Legacy endpoints or new admin routes were added without a global router or dependency injection for authorization.
**Prevention:** All sensitive routes must explicitly require an auth dependency (e.g. `Depends(verify_admin)`) and never rely on frontend obfuscation.
