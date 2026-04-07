## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-03-22 - [LaTeX Injection in PDF Generation]
**Vulnerability:** LaTeX Injection via unsanitized user inputs (e.g., patient name, dict keys) in `backend/latex_report.py` when running `pdflatex`.
**Learning:** Python f-strings in dynamically rendered LaTeX templates can be exploited if input data contains LaTeX control sequences (like `&`, `%`, `\\`). Keys from user-supplied dictionaries like `req.probabilities.keys()` can serve as hidden injection vectors.
**Prevention:** Use a dedicated regex single-pass string-escaping function (`escape_latex`) on all externally sourced inputs before interpolating them into LaTeX templates to ensure special characters are safely mapped to their text equivalents without evaluating recursively.
