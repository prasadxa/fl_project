## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2026-11-01 - LaTeX Injection in Reports
**Issue:** User inputs (e.g., patient name) were directly interpolated into a LaTeX template string.
**Learning:** Unsanitized user inputs in LaTeX templates allow users to execute arbitrary code during document generation.
**Prevention:** Always escape characters with special meaning in LaTeX before interpolation (e.g., \, _, %, $, &, #, ^, ~, {, }).
