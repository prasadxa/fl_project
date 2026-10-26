## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-10-26 - [String Formatting Issue in LaTeX Report Generation]
**Issue:** Missing sanitization of user-controlled inputs within string formatting routines passing data into LaTeX generation blocks in `backend/latex_report.py`.
**Learning:** Constructing document strings (like LaTeX sources) by direct interpolation of user inputs allows injection of arbitrary parsing commands, enabling command execution vulnerabilities.
**Prevention:** Consistently apply domain-specific escaping functions (e.g., escaping special LaTeX characters and neutralising backslashes) to all user-controlled variables before interpolating them into a formatting template.
