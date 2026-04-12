## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-24 - LaTeX Command Injection via PDF Report Generation
**Vulnerability:** The pdflatex report generation directly interpolates unsanitized user inputs (patient name, ID, dict keys) into the .tex document via Python f-strings. This allows an attacker to inject arbitrary LaTeX commands (e.g., \input{/etc/passwd} or \write18 to execute shell commands) which pdflatex evaluates during compilation.
**Learning:** When generating documents via subprocess calls to systems with their own powerful macro languages (like LaTeX or Ghostscript), any dynamic content—including "safe" looking dictionary keys derived from requests—must be rigorously sanitized. A single-pass regex replacement is necessary to prevent sequential replacement bugs.
**Prevention:** Always coerce interpolated variables to strings and escape all special characters used by the target language (for LaTeX: \, &, %, $, #, _, {, }, ~, ^) using a dedicated escaping function before inserting them into templates.
