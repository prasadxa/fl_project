## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2026-10-30 - [Fix issue in PDF Generator]
**Issue:** Unsanitized user inputs are interpolated into LaTeX string templates.
**Learning:** LaTeX special characters can be used to inject malicious LaTeX commands, leading to file read or denial of service.
**Prevention:** Always escape user inputs by replacing special characters (e.g., \, &, %, $, #, _, {, }, ~, ^) before interpolating into LaTeX templates.
