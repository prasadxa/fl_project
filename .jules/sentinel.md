## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-18 - Fix LaTeX Injection Vulnerability in PDF Report Generation
**Vulnerability:** User-supplied inputs (patient name, ID, dob, gender, session ID, timestamp) were inserted directly into an f-string LaTeX template (`backend/latex_report.py`) without escaping, creating a severe command injection vulnerability (e.g., passing `\input{/etc/passwd}` could read arbitrary files or achieve RCE).
**Learning:** Python f-strings evaluating directly into `.tex` templates are highly susceptible to LaTeX injection. A single-pass regular expression replace mechanism using a mapping dictionary ensures that malicious patterns like `\input` are safely transformed (e.g. `\textbackslash{}input`), while escaping all other dangerous characters (like `%`, `&`, `_`, etc.).
**Prevention:** Always coerce user inputs to strings and escape them through a secure string-mapping function before embedding into raw LaTeX documents, especially when calling `subprocess.run(["pdflatex", ...])`. Ensure the escape logic evaluates sequentially avoiding cascading partial-replace bugs.
