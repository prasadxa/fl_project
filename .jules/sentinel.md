## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-31 - LaTeX Injection in PDF Generation
**Vulnerability:** Unsanitized user inputs in f-strings passed to pdflatex allow arbitrary LaTeX command execution.
**Learning:** Dynamically rendered dictionary keys derived from request payloads and arbitrary strings in templates are hidden injection vectors and must be escaped.
**Prevention:** Use a single-pass replacement logic to safely escape LaTeX special characters, ensuring inputs are coerced to strings first.
