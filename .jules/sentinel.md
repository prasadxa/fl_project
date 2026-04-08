## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-04-08 - LaTeX Command Injection in PDF Generation
**Vulnerability:** Unsanitized user inputs in LaTeX template.
**Learning:** Always sanitize inputs before LaTeX interpolation to prevent command injection like `\input{}`.
**Prevention:** Implement and use an `escape_latex` function to escape LaTeX special characters.
