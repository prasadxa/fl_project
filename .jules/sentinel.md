## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-18 - [LaTeX Injection in PDF Report Generation]
**Vulnerability:** LaTeX injection via unescaped user inputs in `backend/latex_report.py`.
**Learning:** Dynamically rendered inputs like patient names, IDs, and dictionary keys must be explicitly escaped before LaTeX template interpolation to prevent malicious payloads from executing arbitrary commands or altering document structure.
**Prevention:** Always use a comprehensive string replacement mapping for special LaTeX characters (e.g., `&, %, $, #, _, {, }, ~, ^, \\`) and coerce all inputs to strings before escaping.
