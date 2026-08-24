## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-03-22 - [Unauthenticated Admin Endpoints]
**Vulnerability:** The /api/admin/* endpoints (stats, feedback, exports, sessions) were completely unauthenticated, exposing potentially sensitive clinical data and metrics to the public internet.
**Learning:** Using `window.open` on the frontend for CSV/Excel exports makes it impossible to attach custom `Authorization` headers. We must use `fetch` + `URL.createObjectURL` to programmatically download the files. Furthermore, Playwright requires custom event listeners (`page.on('dialog')`) to handle native `prompt()` boxes triggered by the frontend.
**Prevention:** Always secure all `/admin` routes using dependency injection in FastAPI (`Depends(verify_admin)`). Use `HTTPBasic` credentials checking securely with `secrets.compare_digest` against securely configured environment variables, never falling back to hardcoded default passwords.
