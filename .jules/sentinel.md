## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-24 - Add authentication to admin endpoint
**Vulnerability:** Admin endpoints lacked authentication, allowing unauthorized access to sensitive statistics and feedback data.
**Learning:** It is crucial to secure administrative endpoints with authentication, even if they are only intended for internal use.
**Prevention:** Always require authentication for endpoints that expose administrative functionality or sensitive data. Do not rely on obscurity or network segmentation alone.
