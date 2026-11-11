## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-11-11 - [LaTeX Injection in PDF Report Generation]
**Issue:** User input (like patient name, ID, etc.) from `ReportRequest` was directly formatted into a LaTeX template string and compiled via `pdflatex` using `subprocess.run` without escaping. An attacker could pass input like `\input{/etc/passwd}` or `\write18{...}` resulting in arbitrary file read or potential Remote Code Execution (RCE).
**Learning:** Directly interpolating user-controlled strings into templates that are subsequently evaluated or compiled by external programs (like `pdflatex` or SQL) can lead to critical injection vulnerabilities if not properly escaped or parameterized.
**Prevention:** Always sanitize or escape user input appropriately for the target execution context before inclusion, or use templating engines that handle contextual escaping automatically. For LaTeX, this involves escaping special characters (`& % $ # _ { } ~ ^ \`).
