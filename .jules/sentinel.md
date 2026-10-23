## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-18 - Missing string escaping in report generator
**Issue:** Unsanitized user input is injected directly into templates in backend/latex_report.py, allowing string formatting abuse.
**Learning:** External data should always be escaped before being placed in structured format documents to prevent unintended parsing.
**Prevention:** Apply dedicated escaping functions for the target format to all external variables prior to interpolation.
