## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-03-24 - [LaTeX Injection in PDF Generation]
**Issue:** User-provided patient data fields were directly embedded into a LaTeX document without escaping, allowing for arbitrary LaTeX command injection (e.g., \immediate\write18 for RCE).
**Learning:** Formatting unescaped user input into LaTeX source code poses a high security risk because LaTeX allows for external shell execution and file reads/writes by design.
**Prevention:** Always sanitize user inputs before placing them in LaTeX documents by escaping special characters, primarily the backslash '\', using a custom substitution function.
