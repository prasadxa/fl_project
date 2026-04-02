## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2026-04-02 - [LaTeX Injection in PDF Report Generation]
**Vulnerability:** User-controlled input (like patient names and IDs) were directly interpolated into a generated `.tex` string in `backend/latex_report.py` without any escaping.
**Learning:** Python f-strings formatting data directly into LaTeX templates creates a high-severity code execution vulnerability, as unescaped input like `\input{/etc/passwd}` or `\write18{...}` can be executed by `pdflatex`. Dictionary keys (like `req.probabilities.keys()`) can also be an injection vector.
**Prevention:** Always parse and explicitly escape all dynamically rendered data via an explicit, single-pass replacement function (`escape_latex`) before it enters a LaTeX f-string, and coerce all inputs to strings to prevent TypeErrors.
