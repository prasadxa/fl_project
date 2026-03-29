## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2025-02-18 - [LaTeX Injection Vulnerability in PDF Report Generation]
**Vulnerability:** Arbitrary LaTeX execution via unsanitized user inputs (e.g., patient name, ID) being interpolated directly into a LaTeX template before pdflatex compilation. This allows attackers to perform command injection (e.g., `\write18`) or arbitrary file read (e.g., `\input{...}`).
**Learning:** Python f-strings or `.format()` do not escape strings by default, making them extremely dangerous for generating structured text formats like LaTeX, HTML, or SQL.
**Prevention:** Always use a single-pass regex replacement function to correctly escape all special characters (like `\`, `{`, `}`, `&`, `%`, `$`, `#`, `_`, `~`, `^`) before interpolating user-supplied strings into a template.
