## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## $(date +%Y-%m-%d) - Unauthenticated Admin Endpoints & Secure Downloads
**Vulnerability:** The /api/admin/* endpoints (stats, feedback, export-csv, export-excel) were fully public without any authentication, exposing all sensitive system statistics, user feedback logs, and bulk CSV/Excel exports.
**Learning:** Adding HTTP Basic Auth backend protection creates a challenge for frontend downloads because standard `window.open` does not attach custom Javascript-based `Authorization` headers. We must migrate direct file URL navigation to programmatic `fetch()` blobs.
**Prevention:** Always verify that internal diagnostic or admin routes are protected by robust authentication middleware (using `secrets.compare_digest` for timing attack safety), and ensure frontend download mechanics natively support auth headers before finalizing the design.
