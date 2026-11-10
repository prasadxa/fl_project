## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-03-24 - [LaTeX Injection in PDF Report Generator]
**Issue:** Unsanitized user inputs (e.g., patient name, ID) were directly interpolated into a LaTeX template for PDF generation, allowing potential arbitrary LaTeX execution.
**Learning:** LaTeX special characters (like \, {, }, _, &, %) can be used by an attacker to alter the document structure, read local files, or execute commands if shell-escape is enabled.
**Prevention:** Always sanitize user-provided text before injecting it into LaTeX documents by escaping special characters and carefully handling backslashes (using placeholders to avoid double escaping).
