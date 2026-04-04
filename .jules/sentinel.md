## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-03-22 - [LaTeX Injection in PDF Report Generation]
**Vulnerability:** Unsanitized user inputs (such as patient names and dynamically rendered dictionary keys derived from request payloads like `req.probabilities.keys()`) were directly interpolated into a LaTeX template string and executed via `pdflatex`, creating a LaTeX injection vulnerability.
**Learning:** Even internal template structures like dictionary keys can be vectors for injection if derived from user payloads. When dynamically building LaTeX templates, all user-provided or payload-derived fields must be explicitly escaped. LaTeX escaping must use a single-pass replacement logic to avoid sequential replacement bugs, and inputs must be coerced to strings to prevent TypeErrors with non-string fields like IDs and datetimes.
**Prevention:** Always sanitize dynamically interpolated strings using a robust LaTeX escaping function before inserting them into a template string.
