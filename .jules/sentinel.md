## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-10-07 - Fix LaTeX Injection in PDF Report Generation
**Vulnerability:** User-provided patient data was injected into a LaTeX template without escaping special characters, leading to potential LaTeX injection (e.g., executing arbitrary commands via `\write18` or reading sensitive files via `\input`).
**Learning:** LaTeX compiles into executable code, and unsanitized input concatenated into templates is equivalent to code injection.
**Prevention:** Always sanitize/escape dynamic input before inserting it into a LaTeX template, taking care to replace backslashes correctly using a placeholder to avoid double escaping.
