## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2026-10-01 - Missing Authentication on Admin Endpoints
**Vulnerability:** The `/api/admin/*` endpoints were accessible without any authentication, exposing sensitive statistics, feedback, and user sessions.
**Learning:** Always implement authentication mechanisms (like HTTPBasic) to protect administrative or sensitive routes, even if they're intended for internal use.
**Prevention:** Ensure all new admin routes include an authentication dependency (`Depends(verify_admin)`) and never hardcode fallback credentials in production logic.
