## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-15 - [Unauthenticated Admin Endpoints]
**Vulnerability:** Admin endpoints for stats, feedback, and exports lacked authentication, allowing unauthorized users to access sensitive aggregate data and feedback logs.
**Learning:** Internal APIs must have explicit authentication layers; never rely on security by obscurity or assume endpoints are only accessible from the frontend.
**Prevention:** Always wrap sensitive endpoints with authentication middleware (e.g., HTTPBasic) and validate credentials securely (e.g., using `secrets.compare_digest` to prevent timing attacks).
