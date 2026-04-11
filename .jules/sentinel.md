## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-04-11 - [LaTeX Injection in Report Generation]
**Vulnerability:** LaTeX injection in `latex_report.py` where unescaped user inputs (like patient name, ID, and dictionary keys) were directly interpolated into a `.tex` template. This could lead to Arbitrary File Read (via `\input{...}`) or RCE (via `\write18` if enabled).
**Learning:** Even dynamically rendered dictionary keys derived from request payloads (like `req.probabilities.keys()`) can serve as hidden injection vectors. Ensure LaTeX escaping uses a single-pass replacement logic to avoid sequential replacement bugs, and always coerce inputs to strings prior to escaping to prevent TypeErrors.
**Prevention:** Always sanitize dynamically interpolated variables in LaTeX templates using a robust, single-pass escaping function that handles all LaTeX special characters (`\`, `&`, `%`, `$`, `#`, `_`, `{`, `}`, `~`, `^`).
