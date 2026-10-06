## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-10-06 - Missing Authentication on Admin Endpoints
**Vulnerability:** The admin dashboard API endpoints (`/api/admin/*`) completely lacked authentication, exposing sensitive system statistics, feedback, and raw diagnostic parameters to the public.
**Learning:** Unauthenticated sensitive endpoints are a critical risk, especially for a medical imaging application where data privacy is paramount. Ensure all administrative routes are protected by authentication middleware.
**Prevention:** Use a dependency like `Depends(verify_admin)` on all sensitive endpoints in FastAPI and use secure comparison (like `secrets.compare_digest`) for password checks.
