## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-18 - Fix LaTeX Injection in PDF Reports
**Vulnerability:** A Command/Code Injection vulnerability existed in `backend/latex_report.py`. User-provided inputs (e.g., patient name, ID) were directly interpolated into a LaTeX template string and passed to `pdflatex` via `subprocess.run()`. An attacker could embed arbitrary LaTeX macros (like `\input{/etc/passwd}` or `\write18`) to exfiltrate files or execute shell commands on the server.
**Learning:** Naively using `.replace()` chains to escape characters in Python strings containing literal backslashes and LaTeX formatting easily leads to double-escaping bugs or incomplete sanitization. Moreover, any string interpolated into a system tool invocation (like `pdflatex`) must be strictly sanitized. Dynamic dictionary keys (like keys of `req.probabilities`) can also be injection vectors.
**Prevention:** Implement a rigorous, single-pass replacement function using `str.maketrans` and `str.translate` that handles all dangerous LaTeX special characters (`\`, `&`, `%`, `$`, `#`, `_`, `{`, `}`, `~`, `^`) at once. Apply this sanitization to *every* external string input before rendering the template.
