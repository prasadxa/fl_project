## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-15 - [LaTeX Injection in PDF Template]
**Vulnerability:** LaTeX injection in `backend/latex_report.py` where unescaped user inputs are inserted into a LaTeX template, potentially leading to arbitrary code execution or PDF compilation failures.
**Learning:** Using raw string formatting (`f""`) for LaTeX templates with unescaped user input opens up LaTeX injection vectors.
**Prevention:** Always escape special LaTeX characters (e.g., `\`, `&`, `%`, `$`, `#`, `_`, `{`, `}`, `~`, `^`) using a robust single-pass escaping function before injecting dynamic data into a LaTeX document.
