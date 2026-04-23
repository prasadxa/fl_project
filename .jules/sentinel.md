## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-18 - Unauthenticated Admin Endpoints
**Vulnerability:** The `/api/admin/*` endpoints in `backend/api.py` were entirely exposed without any authentication, allowing any unauthenticated user to access administrative statistics, download feedback logs, export CSV/Excel reports, and view prediction sessions. This is a critical security risk (missing authentication on sensitive endpoints).
**Learning:** In FastAPI applications, it's common to group administrative routes. Failing to apply a dependency (like `Depends(verify_admin)`) to these endpoints leaves them exposed. The previous structure relied on implicit obscurity rather than explicit security.
**Prevention:** Always secure sensitive routes (like admin panels or data exports) using explicit authentication mechanisms. In FastAPI, use route dependencies (e.g., `dependencies=[Depends(verify_admin)]`) or apply the dependency at the router level for a group of endpoints to ensure no route is accidentally left unsecured.
