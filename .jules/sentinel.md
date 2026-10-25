## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2026-10-25 - LaTeX Injection in PDF Report
**Issue:** Unsanitized user inputs placed directly in LaTeX templates.
**Learning:** LaTeX templates require special escaping distinct from HTML/SQL.
**Prevention:** Apply an escape_latex function to all external inputs placed into .tex files.
