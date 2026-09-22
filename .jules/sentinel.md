## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-18 - LaTeX Injection Vulnerability in PDF Report Generation
**Vulnerability:** User-provided inputs were directly interpolated into a LaTeX template string and compiled via pdflatex, allowing arbitrary LaTeX command injection.
**Learning:** Dynamic generation of LaTeX documents requires strict escaping of special characters, similar to HTML/SQL injection, as LaTeX is Turing-complete and can interact with the file system.
**Prevention:** Always sanitize/escape user inputs using a dedicated LaTeX escape function before inserting them into LaTeX templates, ensuring backslashes and other special symbols are properly neutralized.
