## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-18 - LaTeX Injection in PDF Generator
**Vulnerability:** User inputs (e.g., patient name, ID) and even dictionary keys dynamically rendered from request payloads (e.g. `req.probabilities.keys()`) were interpolated directly into a LaTeX document template without escaping, causing a critical command injection vulnerability via `pdflatex`.
**Learning:** Dictionary keys can serve as hidden injection vectors. Furthermore, when writing a LaTeX escaper, sequential `.replace()` string loops introduce double-escaping bugs (like `\\` becoming `\\textbackslash{}` and then `{` being escaped again). Always use a single-pass replacement logic via translation tables.
**Prevention:** Coerce all dynamically injected content to strings and sanitize using a robust, single-pass `latex_escape` translation table before injecting into LaTeX f-strings.
