## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-09 - Fix LaTeX Injection in PDF Report Generation
**Vulnerability:** The LaTeX PDF generator in `backend/latex_report.py` embedded untrusted user inputs (like patient name, session ID, timestamps, etc) directly into the string templates using f-strings without escaping them, which leads to LaTeX injection. A dictionary generator `req.probabilities.items()` was also embedding unescaped keys.
**Learning:** All values derived directly or indirectly from user inputs must be explicitly escaped in a single pass before being interpolated into string templates, especially when compiling external formats like LaTeX which are vulnerable to command injection.
**Prevention:** Use a dedicated translation-table based `escape_latex` function before rendering string templates. Single pass is important as sequential `.replace` calls may introduce double escaping bugs.
