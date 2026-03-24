## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-03-24 - LaTeX Injection in PDF Report Generator
**Vulnerability:** LaTeX injection vulnerability in `backend/latex_report.py` via unescaped user inputs (e.g. `patient_name`, `patient_id`) when generating the LaTeX template string.
**Learning:** Directly embedding untrusted variables into a LaTeX `.tex` string without escaping special characters (`\`, `{`, `}`, `%`, etc.) allows an attacker to execute arbitrary LaTeX macros. This could potentially read files (`\input{/etc/passwd}`) or crash the LaTeX compilation process.
**Prevention:** Always coerce untrusted variables to strings and apply a single-pass `escape_latex` function using character replacement before interpolating them into a LaTeX template.
