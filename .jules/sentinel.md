## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-24 - Unauthenticated Admin Endpoints
**Vulnerability:** Admin endpoints (`/api/admin/stats`, `/api/admin/feedback`, etc.) were missing authentication, exposing sensitive feedback and session data to unauthenticated users. Also, download endpoints using `window.open()` were bypassing header-based auth.
**Learning:** Even internal or admin-focused APIs must be authenticated explicitly. When securing APIs, consider how file downloads are triggered from the frontend, as `window.open` doesn't pass custom `Authorization` headers.
**Prevention:** Implement an authentication middleware or dependency injection (e.g., `Depends(get_current_admin)`) on all sensitive routes. For frontend downloads from protected routes, use `fetch` with headers, extract the blob, and trigger download via `URL.createObjectURL`. Ensure fallback configurations for missing sensitive environment variables fail securely (e.g., return HTTP 500) rather than using default insecure credentials.
