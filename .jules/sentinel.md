## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-16 - [LaTeX Injection via String Interpolation]
**Vulnerability:** LaTeX injection vulnerability in PDF generation because user input was string-interpolated into a `.tex` template without escaping LaTeX special characters.
**Learning:** Sequential `.replace()` calls can cause double-escaping bugs (e.g., escaping `\` and then escaping the resulting braces `{}`).
**Prevention:** Always use a single-pass translation table (e.g. `str.translate`) to escape all LaTeX control characters simultaneously.
