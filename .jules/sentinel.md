## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2026-09-05 - Secure Admin Authentication
**Vulnerability:** Missing authentication on /api/admin endpoints allowed unauthorized access to sensitive data and statistics.
**Learning:** FastAPI's HTTPBasic requires careful implementation to prevent timing attacks and handle missing configuration safely without falling back to hardcoded defaults.
**Prevention:** Always use `secrets.compare_digest` for credential comparison and ensure environment-based secrets fail securely (500 error) if missing.
