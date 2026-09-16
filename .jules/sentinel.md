## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2026-09-16 - LaTeX Injection Vulnerability in PDF Generation
**Vulnerability:** User-controlled input (like patient name) was interpolated directly into a LaTeX template string before running pdflatex, allowing LaTeX injection which could lead to arbitrary file reads, document manipulation, or command execution.
**Learning:** The pdflatex subprocess requires sanitized input because LaTeX control characters can execute macros or include arbitrary content.
**Prevention:** Always escape special LaTeX characters (such as \, {, }, _, ^, #, &, $, %, ~) in user-controlled inputs before rendering templates.
