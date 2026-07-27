## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-03-22 - [LaTeX Server-Side Template Injection]
**Vulnerability:** The PDF report generation module (`backend/latex_report.py`) interpolates unsanitized user inputs (such as patient names and IDs) directly into a LaTeX template string before compiling it.
**Learning:** Directly embedding inputs into string-formatted LaTeX templates creates a critical Server-Side Template Injection (SSTI) / Arbitrary File Read vulnerability, allowing users to execute commands like `\input{/etc/passwd}`. Python's f-strings are unsafe for code-generation templates if variables contain special LaTeX control characters.
**Prevention:** All user-controlled strings must be passed through a mapping function (like `escape_latex`) to sanitize the 10 core LaTeX reserved characters before they are concatenated into the template.
