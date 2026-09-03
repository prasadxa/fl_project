## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-24 - Missing Authentication on Admin Endpoints
**Vulnerability:** Admin endpoints were exposed without any authentication, allowing unauthorized access to sensitive statistics and data exports.
**Learning:** Important application endpoints require explicit authentication checks, and when relying on environment variables for credentials, the system should fail securely (e.g. 500 error) if they are missing, rather than defaulting to an insecure state.
**Prevention:** Secure all administrative routes using HTTP Basic Authentication or token-based strategies, ensuring configuration defaults are not hardcoded but depend on explicit environment bindings.
