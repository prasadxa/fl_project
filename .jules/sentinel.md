## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-10-24 - [CRITICAL] Fix LaTeX injection vulnerability
**Vulnerability:** Unsanitized user inputs placed directly into a LaTeX template string could allow LaTeX injection.
**Learning:** Always sanitize inputs that are directly interpolated into LaTeX document strings to prevent command execution, formatting disruption, or data exfiltration.
**Prevention:** Use a `sanitize_latex` function to escape LaTeX special characters such as `\`, `&`, `%`, `$`, `#`, `_`, `{`, `}`, `~`, and `^`.
