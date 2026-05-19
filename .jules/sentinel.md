## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-24 - LaTeX Injection in PDF Report Generation
**Vulnerability:** User inputs (like patient name, session ID) were directly interpolated into a LaTeX template string in `backend/latex_report.py` without sanitization. This allowed attackers to inject arbitrary LaTeX commands (LaTeX Injection) which could lead to arbitrary command execution or denial of service when compiled using `pdflatex`.
**Learning:** Using `replace` repeatedly for multiple special characters can cause double-escaping bugs (e.g., escaping `\` first, and then escaping the `\` in the replacement string). A translation table with a single-pass regex replace is much safer and more robust.
**Prevention:** Always sanitize/escape inputs before injecting them into execution templates (LaTeX, Shell, SQL, etc.). Use dedicated encoding libraries or robust, single-pass translation mechanisms like a dictionary with `re.sub` for string escaping to avoid double-escaping.
