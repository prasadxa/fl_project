## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-24 - LaTeX Injection in PDF Report Generation
**Vulnerability:** Unsanitized user inputs (e.g., patient name, session ID) were directly interpolated into a LaTeX template `r.tex` and compiled with `pdflatex`, allowing arbitrary file read via `\input{...}` and potentially arbitrary code execution depending on compiler flags.
**Learning:** System commands or template rendering tools like LaTeX, which support embedded commands or file inclusion, must strictly sanitize inputs if used with untrusted data.
**Prevention:** Implement an `escape_latex` function to replace LaTeX special characters (`\`, `&`, `%`, `$`, `#`, `_`, `{`, `}`, `~`, `^`) with their literal, safe representations before injecting them into templates. Use a unique placeholder for backslashes during substitution to avoid double-escaping.
