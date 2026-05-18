## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-18 - [Hardcoded Basic Auth Fallback Credentials]
**Vulnerability:** Hardcoded administrative credentials used as fallbacks when environment variables are missing.
**Learning:** Supplying default fallback values for critical secrets (like passwords) within the application code directly exposes credentials and violates the 'fail securely' principle.
**Prevention:** When configuring server credentials (e.g., `ADMIN_USER`, `ADMIN_PASS`) via `os.getenv()`, do not use hardcoded fallback values. Instead, verify their existence and explicitly raise an HTTP 500 error if they are unconfigured to ensure a fail-secure architecture.
