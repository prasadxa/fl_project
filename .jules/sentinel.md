## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-24 - LaTeX Injection in PDF Reports
**Vulnerability:** Unescaped user inputs (including dict keys) in LaTeX templates can be exploited for RCE or arbitrary file reads.
**Learning:** Single-pass replacements are crucial to prevent sequential replacement bugs, and dict keys (like probability classes) must also be escaped.
**Prevention:** Coerce all inputs to strings and escape all LaTeX special characters before interpolation.
