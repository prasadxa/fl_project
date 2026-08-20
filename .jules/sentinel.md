## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2025-02-20 - Fix Unprotected Admin Endpoints

**Vulnerability:** The backend endpoints under `/api/admin/*` were completely unauthenticated, exposing sensitive application statistics, user feedback logs, and download capabilities for admin reports.
**Learning:** Even internal or "hidden" administrative endpoints must be explicitly protected. Without authentication, anyone discovering the routes could bypass the UI and extract operational data.
**Prevention:** All sensitive routes must have strict authentication dependencies attached (e.g., `Depends(verify_admin)` in FastAPI). Additionally, when protecting previously unauthenticated frontend functionality like CSV/Excel downloads, `window.open` cannot pass Authorization headers; such logic must be refactored to use standard `fetch` with object URL blob downloads to ensure the headers are sent securely.
