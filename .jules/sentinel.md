## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2025-02-20 - Fix LaTeX injection vulnerability
**Vulnerability:** User inputs (e.g., patient name, ID) were directly interpolated into a LaTeX template in backend/latex_report.py without escaping, allowing for LaTeX injection.
**Learning:** When generating PDF reports via LaTeX, all user-provided strings must be sanitized/escaped to prevent injection of malicious LaTeX commands.
**Prevention:** Always use a dedicated escaping function to sanitize user inputs before embedding them in LaTeX templates.
