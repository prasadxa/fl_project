## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-04-10 - [LaTeX Injection in PDF Generation]
**Vulnerability:** LaTeX injection in `backend/latex_report.py` when dynamically creating PDF reports via a Python f-string. User inputs (patient names, IDs, etc.) containing LaTeX characters (`\`, `{`, `}`, `$`, `&`, `%`, `_`, `^`, `~`, `#`) could execute arbitrary LaTeX commands, potentially causing denial of service or arbitrary file inclusion.
**Learning:** Naively escaping backslashes first in a multi-pass approach might cause double-escaping or re-evaluations. The keys of dictionary payloads from user requests (e.g. `req.probabilities.keys()`) can also be injection vectors.
**Prevention:** Always use a single-pass regex replacement to sanitize dynamic variables before interpolating them into LaTeX templates, and coerce non-string fields to strings to prevent TypeErrors.
