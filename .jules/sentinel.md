## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2025-02-05 - Secured Admin Endpoints & Fetch Interceptors
**Vulnerability:** The `/api/admin/*` endpoints were exposed without any authentication, allowing unauthenticated access to sensitive clinical statistics and feedback logs. Additionally, frontend CSV/Excel download buttons used `window.open`, which bypasses custom headers.
**Learning:** In a federated AI system with legacy plain-text reporting mechanisms, admin endpoints might be neglected while main application endpoints are tightly scoped. Bypassing `window.open` using `fetch()` with `window.URL.createObjectURL` is a critical pattern when protecting previously open static-file-style downloads with custom HTTP headers.
**Prevention:** Always scope admin and export endpoints behind strict authentication dependencies. Do not use `window.open` for protected resource downloads.
