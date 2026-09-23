## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2026-09-23 - [Missing Admin Endpoint Authentication]
**Vulnerability:** Admin endpoints for stats, feedback, and exports lacked authentication, exposing sensitive system data and operations.
**Learning:** Always ensure administrative or internal-only API routes are secured by default. Do not assume obscured URLs provide security.
**Prevention:** Implement explicit authentication guards (e.g., HTTP Basic with environment-driven credentials) on all sensitive routes, failing securely (HTTP 500) if configuration is missing.
