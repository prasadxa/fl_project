## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-10-24 - [PDF Report Input Sanitization]
**Issue:** Missing input sanitization in PDF report generation.
**Learning:** When interpolating user-controlled strings into dynamically generated documents, characters with special syntax meaning must be properly escaped.
**Prevention:** Always apply an escaping function to user inputs before interpolation into formatted document templates.
