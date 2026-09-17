## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-18 - Prevent LaTeX Injection in PDF Report Generation
**Vulnerability:** User-provided patient data is directly interpolated into a LaTeX template string in `backend/latex_report.py` without any escaping.
**Learning:** Malicious inputs containing LaTeX control characters can break PDF generation or execute arbitrary LaTeX commands (LaTeX injection).
**Prevention:** Always sanitize user-controlled inputs by escaping special LaTeX characters (such as `\`, `{`, `}`, `_`, `^`, `#`, `&`, `$`, `%`, `~`) before interpolating them into LaTeX templates, ensuring backslashes are replaced first.
