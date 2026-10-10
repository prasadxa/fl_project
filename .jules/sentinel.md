## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-10-10 - [LaTeX Injection in PDF Generation]
**Vulnerability:** Arbitrary LaTeX injection via unsanitized user inputs (like patient name) in the PDF generation template, allowing Remote Code Execution and Local File Inclusion.
**Learning:** Sequential string replacement for escaping LaTeX must be done carefully. For example, backslashes must be replaced with a placeholder devoid of underscores to prevent accidental double-escaping when underscores are replaced later.
**Prevention:** Always sanitize/escape all user-provided data before embedding it into LaTeX templates using a robust escaping sequence.
