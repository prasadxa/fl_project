## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-23 - LaTeX Injection in PDF Report Generator
**Vulnerability:** User-controlled inputs (`patient_name`, `session_id`, etc.) were directly injected into a LaTeX template string evaluated by `pdflatex`, allowing arbitrary file reading and code execution via commands like `\input{/etc/passwd}`.
**Learning:** LaTeX templates are highly susceptible to injection if user input is not escaped. Simple backslash replacement is insufficient and can lead to double escaping.
**Prevention:** Always sanitize user inputs before placing them in LaTeX templates. Use a dedicated escaping function that escapes backslashes using a placeholder, escapes other special characters (`#`, `$`, `%`, `&`, `_`, `{`, `}`, `^`, `~`), and finally replaces the placeholder with `\textbackslash{}`.
