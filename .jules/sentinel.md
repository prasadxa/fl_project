## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2026-04-25 - [Missing Authentication on Admin Endpoints]
**Vulnerability:** Admin endpoints (/api/admin/*) were fully exposed without any authentication.
**Learning:** Always explicitly protect administrative endpoints with authentication middleware or dependencies, rather than relying on UI-level hiding.
**Prevention:** Ensure new endpoints under /api/admin/ automatically inherit an admin-role dependency.
