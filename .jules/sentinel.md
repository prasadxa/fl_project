## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2025-02-14 - Admin Endpoint Authentication Bypass
**Vulnerability:** The backend `/api/admin/*` endpoints, which expose sensitive aggregated statistics, raw clinician feedback, and detailed API session logs, were completely unauthenticated. Anyone could query these endpoints and download CSV/Excel exports of internal system data. The frontend also did not handle or enforce authentication for the admin route.
**Learning:** High-value aggregated data endpoints are just as sensitive as direct PII and must be explicitly protected by default. Relying purely on obscurity (e.g., hiding the link) is not a security strategy. When implementing quick HTTP Basic authentication to secure previously open endpoints, the environment configuration must fail-secure (e.g., throwing a 500 error if secrets are missing) rather than failing open or using default fallback passwords which can be easily guessed. Additionally, string comparisons for credentials must use `secrets.compare_digest()` to prevent timing attacks.
**Prevention:**
1. Always inject authentication dependencies into internal or administrative API routes by default during creation.
2. Require environment-injected credentials to be validated at runtime startup or dependency execution, crashing the application (or specific route) with a 500 status if they are unconfigured.
3. When refactoring frontend file downloads to support custom headers (like Authorization), avoid `window.open` and rely on `fetch` with programmed Blobs.
