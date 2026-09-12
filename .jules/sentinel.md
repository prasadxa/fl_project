## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-03-22 - [Missing Authentication on Admin Endpoints]
**Vulnerability:** Admin endpoints (`/api/admin/*`) lack authentication and authorization logic, exposing sensitive statistical, prediction and feedback data to any user.
**Learning:** FastAPI's default dependency injection makes securing multiple endpoints easy, but custom implementations using `HTTPBasic(auto_error=False)` must include `headers={"WWW-Authenticate": "Basic"}` manually on 401 raises for native browsers/clients to intercept. In Python, use `secrets.compare_digest` instead of `==` for comparing sensitive strings to prevent timing attacks.
**Prevention:** Make sure all endpoints serving sensitive administrative functionalities have an authorization guard (e.g. `Depends()`) attached before pushing to production. Do not provide fallback hardcoded credentials if environment variables are missing; fail securely.
