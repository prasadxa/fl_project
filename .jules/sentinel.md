## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-06-28 - Missing Authentication on Admin Endpoints
**Vulnerability:** The backend administrative endpoints (`/api/admin/stats`, `/api/admin/feedback`, etc.) were completely unauthenticated, allowing any unauthenticated user to access potentially sensitive aggregate data and feedback logs. Additionally, the frontend implemented downloads using `window.open`, which bypasses customized fetch headers such as `Authorization`.
**Learning:** Admin routes must have authentication strictly enforced at the routing level (e.g., using `Depends` in FastAPI). In addition, when implementing HTTP Basic Authentication, it's crucial to "fail securely": if required environment variables (like `ADMIN_USER` and `ADMIN_PASS`) are not set, the application should throw an HTTP 500 Server Error to prevent accidental open access. For frontend downloads requiring authentication, programmatically fetching the blob and using `URL.createObjectURL` is necessary to pass the required headers.
**Prevention:**
- Explicitly configure and enforce dependency injection for authentication (e.g., `Depends(get_current_admin)`) on all sensitive routes.
- Validate the presence of authentication environment variables on startup or within the dependency and raise a 500 error if they are missing.
- Use `secrets.compare_digest` to mitigate timing attacks when validating basic auth credentials.
- Handle protected file downloads in the frontend using `fetch` and `URL.createObjectURL` rather than `window.open()`.
