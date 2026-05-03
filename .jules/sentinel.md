## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-03 - [Fix Missing Auth on Admin Endpoints]
**Vulnerability:** The /api/admin/* endpoints lacked authentication, exposing sensitive data.
**Learning:** Administrative endpoints should always enforce authentication checks before returning data.
**Prevention:** Ensure all new admin routes use the FastAPI Depends(verify_admin) dependency.
