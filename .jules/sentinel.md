## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-24 - [Document Format Corruption]
**Issue:** Unsanitized user inputs (`patient_name`, `patient_id`, `date_of_birth`, `gender`) are directly interpolated into document templates in `backend/latex_report.py`.
**Learning:** Directly embedding untrusted input into templates without escaping special characters allows document formatting disruption.
**Prevention:** Always sanitize inputs interpolated into templates by escaping characters like `\\`, `&`, `%`, `$`, `#`, `_`, `{`, `}`, `~`, `^`.
