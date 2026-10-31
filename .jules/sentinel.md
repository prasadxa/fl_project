## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-10-31 - LaTeX Template Generation Bug
**Issue:** Missing sanitization for LaTeX template variables (e.g. `req.patient.patient_name`) leading to potential code execution.
**Learning:** Generating raw LaTeX files by interpolating unsanitized strings allows special characters to be interpreted as commands.
**Prevention:** Escape LaTeX special characters (like backslashes, ampersands, percentages) sequentially, using a placeholder for backslashes to prevent double-escaping.
