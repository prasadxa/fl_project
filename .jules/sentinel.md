## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-10-08 - LaTeX Injection in PDF Generator
**Vulnerability:** User-controlled strings (`patient_name`, `session_id`, etc.) were directly formatted into a LaTeX template in `backend/latex_report.py` without escaping. This allows LaTeX injection (e.g., `\input{/etc/passwd}` or `\write18`), leading to arbitrary file read or command execution during the `pdflatex` compilation step.
**Learning:** Any dynamic values inserted into LaTeX templates must be meticulously escaped, as the TeX engine acts as an interpreter capable of system interactions.
**Prevention:** Implement and enforce a comprehensive LaTeX escaping function for all untrusted inputs before inserting them into `.tex` templates.
