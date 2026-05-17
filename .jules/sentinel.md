## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-17 - Secure Admin Endpoints with HTTPBasic Auth
**Vulnerability:** The `/api/admin/*` endpoints were exposed without any authentication, allowing unauthorized access to statistics, feedback, and session records.
**Learning:** Admin routes need protection using `fastapi.security.HTTPBasic`. Using `secrets.compare_digest` is crucial when validating credentials to prevent timing attacks.
**Prevention:** Always ensure new administrative routes in `api.py` are decorated with `dependencies=[Depends(verify_admin)]`.
