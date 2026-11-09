## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2026-11-09 - [High] Fix template rendering issue
**Issue:** Unsanitized user input in LaTeX template generation leading to potential remote code execution via LaTeX injection.
**Learning:** LaTeX templates generated using Python f-strings must escape special characters to prevent command injection.
**Prevention:** Use a dedicated sanitization function to escape LaTeX special characters (\, &, %, $, #, _, {, }, ~, ^) before rendering user input in LaTeX documents.
