## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2025-02-14 - Prevent LaTeX Injection in PDF Report Generation
**Vulnerability:** The `build_latex_report` function in `backend/latex_report.py` directly interpolates user-provided text (like patient name, patient ID, date of birth, etc.) into a LaTeX string without escaping LaTeX special characters. This allows LaTeX injection attacks, where a malicious user could supply payloads (like `\input{/etc/passwd}` or `\write18{...}`) to read server files, execute arbitrary commands (if `shell_escape` is enabled), or simply break PDF generation.
**Learning:** Raw string interpolation for template engines, particularly those running system binaries like LaTeX, is highly dangerous. Input parameters must be sanitized and escaped to treat special symbols strictly as text.
**Prevention:** Always escape sensitive LaTeX control characters (e.g., `\`, `&`, `%`, `$`, `#`, `_`, `{`, `}`, `~`, `^`) in user-supplied strings before inserting them into a `.tex` template.
