## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-03-24 - [Unauthenticated Admin Endpoints & Secure Secret Comparison]
**Vulnerability:** The `/api/admin/*` endpoints were exposed without authentication, allowing anyone to access aggregate statistics, feedback logs, prediction sessions, and download full feedback CSV/Excel reports.
**Learning:** Admin routes need explicit protection in FastAPI, which wasn't initially present. When implementing basic authentication with environment variables, `secrets.compare_digest` must be used instead of standard string equality (`==`) to compare credentials in order to prevent timing attacks.
**Prevention:** Always ensure sensitive endpoints are protected by an authentication dependency. Use `secrets.compare_digest` when verifying secrets like passwords or API keys.
