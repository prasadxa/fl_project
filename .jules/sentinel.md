## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2026-09-26 - [LaTeX Injection Fix]
**Vulnerability:** LaTeX Injection via unescaped patient details
**Learning:** Unescaped strings interpolated into LaTeX templates can cause DoS or arbitrary file read during compilation.
**Prevention:** Use an escape function that replaces backslashes with a unique placeholder before escaping other characters to prevent double escaping.
