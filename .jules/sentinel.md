## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-03-24 - Fix LaTeX Injection
**Vulnerability:** LaTeX Injection via unsanitized user inputs in PDF reports.
**Learning:** When using pdflatex to generate reports from f-string templates, direct interpolation of user inputs like patient details allows LaTeX code execution. Escaping must be done using a single-pass translation table, and dictionary keys must also be escaped.
**Prevention:** Always escape user-controlled inputs with a dedicated single-pass LaTeX escaping function before interpolating them into .tex templates.
