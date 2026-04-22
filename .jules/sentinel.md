## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-04-22 - [LaTeX Injection in PDF Report Generation]
**Vulnerability:** Arbitrary file read and potential RCE via LaTeX injection in f-string template used for PDF report generation. Unsanitized user inputs (like patient names and dictionary keys) were directly interpolated into a LaTeX document that was then compiled with pdflatex.
**Learning:** Dynamically generated LaTeX files using f-strings are highly vulnerable to injection attacks. Even seemingly benign inputs or keys derived from request payloads can serve as hidden injection vectors.
**Prevention:** Always mitigate by sanitizing all interpolated inputs using a robust, single-pass replacement function (e.g., via translation tables) to escape special LaTeX characters. Ensure literal double backslashes in Python f-strings are preserved to prevent evaluation errors.
