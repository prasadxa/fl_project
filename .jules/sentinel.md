## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2025-02-27 - [Missing Authentication on Admin API Endpoints]
**Vulnerability:** The FastAPI backend exposed sensitive `/api/admin/*` endpoints without any authentication, allowing unrestricted access to internal statistics, feedback data, and exports.
**Learning:** The endpoints were not secured by `HTTPBasic` credentials despite the environment theoretically requiring `ADMIN_USER` and `ADMIN_PASS`. Also learned that it's important to use `secrets.compare_digest` to prevent timing attacks.
**Prevention:** Always verify that all admin/sensitive endpoints explicitly declare authentication dependencies (e.g., `Depends(get_current_admin)`) and confirm frontend functions are injecting the proper headers.
