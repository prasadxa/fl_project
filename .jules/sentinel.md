## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-11-06 - [Latex sanitization]
**Issue:** Unsanitized user inputs in LaTeX template.
**Learning:** Direct injection of user input into string templates passed to pdflatex can lead to arbitrary code execution.
**Prevention:** Always sanitize or escape user inputs before embedding them into LaTeX templates.
