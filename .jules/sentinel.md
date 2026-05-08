## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2026-05-08 - Fix LaTeX Command Injection
**Vulnerability:** LaTeX command injection in `backend/latex_report.py` caused by rendering unsanitized variables (like `req.patient.patient_name`) in `pdflatex` diagnostic report generation.
**Learning:** Sequential `str.replace()` loops on dynamic template strings can lead to double escaping (where an escaped backslash becomes vulnerable again). A single-pass character substitution using a dictionary and translation tables is required.
**Prevention:** Coerce all dynamically rendered strings into LaTeX-escaped templates using a single-pass sanitizer before writing the `.tex` file.
