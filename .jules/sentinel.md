## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-24 - LaTeX Injection Vulnerability in Diagnostics Report
**Vulnerability:** LaTeX injection vulnerability in `backend/latex_report.py` due to raw string interpolation of user-controlled inputs (e.g. patient name, ID, probabilities keys) into `.tex` templates.
**Learning:** `pdflatex` can execute arbitrary commands (`\write18`) or read local files if malicious inputs contain unescaped LaTeX control characters (like `\`, `{`, `}`). Unsanitized user data in medical reporting logic poses a critical local file read and remote code execution risk.
**Prevention:** Always sanitize inputs interpolated into LaTeX by replacing special characters (like `\`, `%`, `$`, `&`, `_`, `{`, `}`) with their escaped equivalents (`\textbackslash{}`, `\%`, etc.). Ensure dictionaries generated from user requests (e.g., probability keys) are also sanitized before interpolation. Coerce variables to strings prior to escaping to avoid TypeErrors.
