## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-24 - Fix LaTeX Injection Vulnerability in Report Generation
**Vulnerability:** Unsanitized user inputs (e.g., `req.patient.patient_name`) were directly interpolated into a LaTeX string in `backend/latex_report.py`, enabling LaTeX injection which could allow arbitrary file reads or command execution.
**Learning:** Even non-HTML templates like LaTeX are susceptible to injection attacks if user data isn't properly escaped before compilation.
**Prevention:** Always apply context-specific escaping (e.g., escaping `& % $ # _ { } ~ ^ \`) to all user inputs embedded in LaTeX strings before running `pdflatex`.
