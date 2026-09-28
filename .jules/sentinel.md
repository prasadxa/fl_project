## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-03-24 - [LaTeX Injection in PDF Reports]
**Vulnerability:** LaTeX injection vulnerability in `backend/latex_report.py` where unsanitized user inputs (patient name, ID, etc.) were interpolated directly into a LaTeX string template.
**Learning:** Passing unsanitized user inputs directly into `f-strings` intended for LaTeX allows attackers to inject malicious LaTeX commands, such as `\input{}` to read local files or `\write18{}` for command execution.
**Prevention:** Always escape special LaTeX characters (e.g., `\, {, }, &, %, $, #, _, ~, ^`) in user-provided strings before inserting them into a LaTeX template. Pay special attention to sequential replacements to avoid double-escaping.
