## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-24 - Document Generation Sanitization
**Issue:** LaTeX injection vulnerability in `backend/latex_report.py` during document generation.
**Learning:** When generating documents by interpolating untrusted input into LaTeX templates, failure to sanitize user inputs leads to arbitrary code execution or document manipulation. `pdflatex` compilation can be manipulated by malicious inputs like `\input{/etc/passwd}` or similar.
**Prevention:** Always implement a comprehensive context-aware escaping function for LaTeX control characters (e.g., `\`, `&`, `%`, `$`, `#`, `_`, `{`, `}`, `~`, `^`) before inserting user inputs into LaTeX templates. Specifically escape backslashes first using a placeholder to prevent double escaping.
