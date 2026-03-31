## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-18 - [LaTeX Injection in PDF Generation]
**Vulnerability:** LaTeX injection in 'backend/latex_report.py' allowed unescaped user-controlled patient data and probabilities keys to execute arbitrary pdflatex code, potentially leading to command execution or file read.
**Learning:** User inputs dynamically interpolated into LaTeX templates must be explicitly escaped, including dictionary keys. Single-pass replacement logic should be used to avoid sequential replacement bugs, and inputs must be coerced to strings before escaping.
**Prevention:** Always use a comprehensive 'tex_escape' function for all dynamic insertions in LaTeX templates and coerce inputs to strings.
