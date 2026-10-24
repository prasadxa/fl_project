## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2026-10-24 - String Escaping in Document Generation
**Issue:** String templates without escaping can lead to unintended macro execution when passed to document generators (like LaTeX) using user input.
**Learning:** String interpolation mechanisms (f-strings) should not directly receive unsanitized user inputs, especially when generating formats that support arbitrary code execution.
**Prevention:** Always implement explicit sanitization/escaping layers (like mapping `\` and special symbols like `&` and `%`) before injecting variables into string templates for parsers.
