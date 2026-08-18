## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-10-27 - [CRITICAL] Unauthenticated Admin Endpoints
**Vulnerability:** Admin endpoints under `/api/admin/*` were completely unauthenticated and accessible to any user.
**Learning:** FastAPI endpoints without explicit `Depends` security checks are exposed by default, even if logically grouped as "admin". Relying on front-end obfuscation is insufficient. When securing endpoints, any file download endpoints using `window.open` must also be refactored to use authenticated `fetch` and blob URLs.
**Prevention:** Always implement robust backend authentication mechanisms (e.g., HTTP Basic Authentication or token-based) and explicitly inject them as dependencies on all sensitive routes. Fail securely (HTTP 500) if required credential variables are absent, and use `secrets.compare_digest` to mitigate timing attacks.
