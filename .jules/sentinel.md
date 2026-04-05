## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-03-22 - [LaTeX Injection in PDF Generation]
**Vulnerability:** LaTeX Injection via unescaped string interpolation into `.tex` templates.
**Learning:** Python f-strings or direct concatenation into `.tex` templates allows an attacker to inject LaTeX commands (e.g., via malicious patient names, dictionary keys like `req.probabilities.keys()`), potentially leading to arbitrary file read (using `\input` or `\include`) or code execution if LaTeX is misconfigured. When writing the `escape_latex` function, care must be taken to escape backslashes first or via a single-pass regex to avoid double-escaping.
**Prevention:** Always use a single-pass regex replacement function to sanitize untrusted input before rendering it in LaTeX templates.
