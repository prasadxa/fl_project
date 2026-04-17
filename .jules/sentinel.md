## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-03-22 - [pdflatex Injection Vulnerability via Unescaped Interpolation]
**Vulnerability:** LaTeX injection in `backend/latex_report.py` caused by interpolating unescaped user-controlled inputs (like patient names and probabilities keys) directly into a LaTeX template string using Python f-strings.
**Learning:** Functions executing shell commands like `pdflatex` interpret specific characters as control sequences (e.g., `\` for commands, `{}` for scoping), which can lead to template manipulation, potential file reads, or command injection if unsanitized user data is embedded directly into the template.
**Prevention:** Use a single-pass character escaping logic (e.g., `str.maketrans` and `str.translate`) to securely escape special LaTeX characters (`\`, `{`, `}`, `_`, `^`, `~`, `%`, `$`, `#`, `&`) in all user-controlled inputs before injecting them into LaTeX templates. Avoid consecutive `.replace()` loops as they can cause double-escaping bugs.
