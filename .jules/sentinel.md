## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-08-15 - [Missing Authentication on Admin Endpoints]
**Vulnerability:** /api/admin/* endpoints were exposed without authentication, allowing unauthenticated attackers to access sensitive dashboard statistics and download admin reports exposing potentially sensitive records.
**Learning:** Default backend API routes must explicitly implement secure authentication. Frontend fetch wrappers need to support credential passing and interactive challenge response (401 prompts) when consuming authenticated API endpoints. Use `secrets.compare_digest` in Python rather than standard string equality (`==`) to compare credentials securely and prevent timing attacks. Never hardcode fallback values for sensitive environment variables (e.g., `ADMIN_USER` and `ADMIN_PASS`); let the server fail securely if they are not set. When triggering file downloads from endpoints protected by custom authorization headers (e.g., HTTPBasic), do not use `window.open()`, as it cannot pass custom headers. Instead, use `fetch()` with the authorization headers, extract the response blob, and programmatically trigger the download.
**Prevention:** Implement `fastapi.security.HTTPBasic` on backend sensitive endpoints.
