## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2025-02-28 - Missing Authentication on Admin Endpoints
**Vulnerability:** The `/api/admin/*` endpoints in `backend/api.py` were exposing sensitive statistical data, feedback, and user sessions without any authentication.
**Learning:** Found a missing authorization gap in the backend architecture. All endpoints under the admin path need to be protected.
**Prevention:** Implemented HTTP Basic Authentication utilizing `fastapi.security.HTTPBasic` and `Depends`. We also verified to use `secrets.compare_digest` instead of simple `==` for comparing credentials to prevent timing attacks and ensuring failure if the env vars are missing.
