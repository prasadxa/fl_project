## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-18 - [Missing LaTeX Escaping in Report Generation]
**Issue:** Unsanitized user inputs (patient name, ID, etc.) were placed directly into a LaTeX document template.
**Learning:** Generating documents by directly interpolating strings into template languages like LaTeX allows formatting characters (e.g., \, %, _) to be interpreted as code, leading to compilation failures or unintended formatting.
**Prevention:** Always use a helper function to escape template-specific formatting characters before interpolating user input.
