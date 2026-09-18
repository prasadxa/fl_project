## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-09-18 - [Fix LaTeX injection in PDF report generation]
**Vulnerability:** User-controlled input (e.g., patient name, ID, DOB, gender, timestamp, session ID) was passed directly into a LaTeX template string without escaping, leading to a LaTeX injection vulnerability.
**Learning:** When generating documents from templates like LaTeX, all user-provided data must be properly escaped to prevent arbitrary command execution or document corruption. Backslashes must be handled carefully (e.g., using a placeholder) to avoid double-escaping.
**Prevention:** Always use a dedicated escaping function for the specific template language (e.g., `escape_latex`) before embedding user input into the template.
