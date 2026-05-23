## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2025-02-28 - Add authentication to admin endpoint
**Vulnerability:** Missing authentication on the sensitive `/api/admin/*` endpoints in the FastAPI backend.
**Learning:** Even internal or admin endpoints must be secured by default. The fail-secure approach (raising a 500 error when credentials are not configured) is a robust pattern to prevent accidental exposure of administrative features. Using `secrets.compare_digest()` is critical to prevent timing attacks during credential verification.
**Prevention:** Always implement an authentication scheme (like `HTTPBasic`) and use dependency injection (e.g., `Depends`) on all sensitive routes from the inception of the API.
