## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-06-12 - [Unauthenticated Admin Endpoints]
**Vulnerability:** Admin API endpoints (`/api/admin/*`) and file downloads exposed sensitive application data without authentication.
**Learning:** Endpoints meant for administrative access must not rely solely on obscure routing or frontend obscurity. Downloading files from protected endpoints using `window.open` is impossible without exposing tokens in the URL.
**Prevention:** Implement HTTP Basic Authentication (or a stronger mechanism) across all sensitive API routes. For authenticated file downloads from the frontend, use `fetch` with the necessary `Authorization` headers, extract the blob, and use `URL.createObjectURL(blob)` to trigger the download securely.
