## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2025-05-18 - Fix LaTeX injection in PDF report generation
**Vulnerability:** The application generated LaTeX PDFs dynamically via `pdflatex` using unescaped user-supplied metadata (`patient_name`, `date_of_birth`, etc.) inside a format string. This allowed for LaTeX injection, which could crash the compiler or execute arbitrary system commands if shell-escape was enabled.
**Learning:** `subprocess.run` calls to system compilers like LaTeX are vulnerable to injection even if they are just reading unescaped input from a file. Interpolating user strings securely requires escaping domain-specific syntax logic (e.g. `\`, `{`, `}`, `&`, `%`, `$`, `#`, `_`, `~`, `^`).
**Prevention:** Always escape domain-specific syntax control characters using a single-pass `str.translate` dictionary logic before passing values into templated outputs that will be executed or compiled. Sequential string replacements like `.replace()` can lead to double-escaping vulnerabilities.
