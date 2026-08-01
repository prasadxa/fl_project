## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-03-22 - [FastAPI Admin Authentication]
**Vulnerability:** Admin endpoints (`/api/admin/*`) were unauthenticated and allowed public access to sensitive patient feedback data and diagnostic statistics.
**Learning:** Hardcoding administrative routing under a common path prefix is insufficient; the backend explicitly needs an authentication dependency added to each route decorator using `Depends()`. Furthermore, implementing custom authorization on endpoints triggered by `window.open()` (e.g. for CSV downloads) fails because the browser doesn't send custom headers; we had to manually `fetch()` the endpoints and simulate a file download using `URL.createObjectURL(blob)`.
**Prevention:** Always require authentication on admin endpoints. When building file download endpoints that require authorization headers, do not rely on `window.open()`; use `fetch()` and programmatically trigger downloads.
