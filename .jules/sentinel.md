## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2026-10-16 - LaTeX Injection in PDF Reports
**Vulnerability:** User-provided patient data was injected unescaped into LaTeX f-strings in backend/latex_report.py, allowing arbitrary LaTeX command execution via pdflatex.
**Learning:** LaTeX templates generated dynamically must escape special characters (backslash, underscore, ampersand, etc.) to prevent LaTeX injection.
**Prevention:** Use a dedicated escape function replacing backslashes with a unique placeholder before escaping other characters to avoid double-escaping, then replacing the placeholder.
