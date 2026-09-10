## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-19 - Missing Authentication on Admin Endpoints
**Vulnerability:** The /api/admin/* endpoints lacked authentication, allowing unauthorized users to access aggregate statistics, download feedback CSVs, and view prediction sessions.
**Learning:** Even internal or admin-focused endpoints exposed on an API without explicit authentication logic are vulnerable to unauthorized access. Relying on frontend routing obscurity is insufficient.
**Prevention:** Always implement strong authentication (e.g., HTTP Basic Auth or Bearer tokens) natively on the backend routes for any sensitive endpoints, ensuring failed checks fail securely (e.g., HTTP 401).
