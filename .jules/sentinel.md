## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-18 - [LaTeX Injection Vulnerability in Report Generation]
**Vulnerability:** Arbitrary user input from the API (like patient name) is interpolated directly into a LaTeX template and passed to 'pdflatex' without escaping.
**Learning:** LaTeX compilers evaluate inputs as code, allowing arbitrary command execution (e.g., via '\write18'), file reads ('\input{/etc/passwd}'), and denial of service. It is critical to single-pass escape special characters ('&', '%', '$', '#', '_', '{', '}', '~', '^', '\') for all external inputs.
**Prevention:** Always coerce external inputs to a string and safely escape LaTeX special symbols before inserting them into '.tex' templates.
