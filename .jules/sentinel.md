## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2025-05-18 - LaTeX Injection Vulnerability in PDF Report Generation
**Vulnerability:** User-provided inputs (like patient name, ID, and inferred scan metrics) were directly interpolated into an f-string generating a `.tex` file without any sanitization in `backend/latex_report.py`. An attacker could inject malicious LaTeX commands (e.g., `\write18` if enabled, or arbitrary file reads like `\input{/etc/passwd}`) leading to Remote Code Execution (RCE) or sensitive information disclosure.
**Learning:** Even when interpolating strings into intermediary formats like LaTeX, strict sanitization is crucial. Relying on simple `.replace()` chains often introduces bugs like double-escaping (e.g. `\` to `\textbackslash{}` and then `{` to `\{`).
**Prevention:** Always use a single-pass `str.maketrans` translation table to securely map all relevant LaTeX special characters (`\`, `{`, `}`, `_`, `^`, `#`, `&`, `$`, `%`, `~`) to their safe `\text...` or escaped equivalents before formatting them into `.tex` templates.
