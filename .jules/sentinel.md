## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2025-05-16 - Prevent LaTeX injection with single-pass character mappings
**Vulnerability:** LaTeX diagnostic reports were vulnerable to injection because dynamically interpolated user inputs (like patient names and AI probability keys) weren't escaped before being passed to `pdflatex` via f-strings.
**Learning:** Using sequential `.replace()` calls to escape LaTeX characters can lead to double-escaping bugs (e.g. replacing `\` with `\textbackslash{}` and then inadvertently escaping the `{` again). Furthermore, dynamic dictionary keys mapped into templates also require escaping.
**Prevention:** Always use a single-pass character replacement strategy like Python's `str.maketrans()` with `str.translate()` for robust escaping of multiple special characters, and actively sanitize *both* keys and values derived from dynamic request payloads before interpolation into templates.
