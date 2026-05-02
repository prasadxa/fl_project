## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-10-18 - [Fix LaTeX Injection in PDF Report]
**Vulnerability:** LaTeX Injection via unsanitized user inputs interpolated into a LaTeX template string in `backend/latex_report.py`.
**Learning:** Python f-strings evaluate literal variables directly. When these variables contain LaTeX special characters like `%`, `&`, or `_`, `pdflatex` can interpret them as commands, leading to PDF generation failures or arbitrary LaTeX execution. Sequential string `.replace()` loops can cause double-escaping bugs.
**Prevention:** Use a true single-pass replacement logic via `str.maketrans` and `translate` to escape all LaTeX control characters (`&`, `%`, `$`, `#`, `_`, `{`, `}`, `~`, `^`, `\`) before interpolation.
