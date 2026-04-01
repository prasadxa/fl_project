## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-03-23 - [LaTeX Command Injection in PDF Reports]
**Vulnerability:** LaTeX command injection due to unescaped user inputs in `backend/latex_report.py`. The dynamically rendered LaTeX document (`r.tex`) directly interpolated variables from the `req` payload without sanitization, leading to an RCE vulnerability when executed via `subprocess.run(["pdflatex", ...])`.
**Learning:** Any dynamic rendering process executed via system-level binaries (like `pdflatex`) is susceptible to injection. Variables derived from request payloads must be escaped, particularly in format strings or dynamically rendered keys.
**Prevention:** Implement a strict, single-pass escaping mechanism for all inputs before injecting them into templates intended for command-line tools. Coerce inputs to strings to avoid type-related errors during processing.
