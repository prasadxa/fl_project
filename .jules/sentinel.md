## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-24 - Missing Admin Authentication
**Vulnerability:** Admin endpoints for exporting sensitive feedback and sessions data were completely unauthenticated.
**Learning:** Relying solely on obscure URLs or frontend obfuscation leaves backend endpoints fully exposed to direct API enumeration. Environment variables must be strictly enforced without default fallbacks to ensure secure deployment.
**Prevention:** Always wrap administrative endpoints with an explicit authentication dependency (e.g., HTTPBasic via Depends) and fail securely (HTTP 500) if credential environment variables are missing.
