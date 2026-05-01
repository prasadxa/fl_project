## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-18 - Fix LaTeX Injection in PDF Generator
**Vulnerability:** LaTeX injection in `backend/latex_report.py` allowed arbitrary LaTeX execution via unescaped user inputs (like patient name or dictionary keys) interpolated into the `.tex` template.
**Learning:** `pdflatex` can be exploited to execute shell commands or read files if malicious input isn't sanitized. Typical `.replace()` chains often suffer from double-escaping (e.g. `\` -> `\\` -> `\textbackslash{}`). Dynamically rendered keys like `req.probabilities.keys()` are also vulnerable vectors.
**Prevention:** Use a single-pass `str.translate` with `str.maketrans` to escape all LaTeX special characters at once. Ensure all dynamically generated strings and dictionary keys are also sanitized before interpolation into the template.
