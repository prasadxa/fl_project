## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-24 - [LaTeX Injection]
**Vulnerability:** LaTeX Injection allowing arbitrary file reads and potential RCE.
**Learning:** Directly interpolating user-controlled strings (like patient names) into LaTeX templates (`.tex` files) evaluated by `pdflatex` allows attackers to inject control commands like `\input{/etc/passwd}` or `\write18{...}`.
**Prevention:** Always escape standard LaTeX control characters (`\`, `{`, `}`, `_`, `^`, `#`, `&`, `$`, `%`, `~`) in user-controlled data before embedding them in LaTeX templates. Make sure to escape backslashes first to avoid double-escaping.
