## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-20 - [Template escaping]
**Issue:** Missing input escaping in template generator allows formatting breakout.
**Learning:** Direct string interpolation of user input into templates without proper escaping functions leads to formatting breakouts.
**Prevention:** Always use a dedicated escaping function for the target format before interpolation.
