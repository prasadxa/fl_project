## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.
## 2024-05-18 - LaTeX Format String Injection Fixed
**Issue:** Unsanitized user inputs embedded directly into LaTeX templates (`req.patient.patient_name` etc) using Python f-strings, enabling LaTeX injection attacks (e.g. reading files with `\input{/etc/passwd}`).
**Learning:** When generating documents from templates via string formatting, any field directly sourced from the user (such as patient names and IDs) is an injection vector for the templating engine. The LaTeX compiler executes commands present in the input if they are not escaped.
**Prevention:** Apply a dedicated `escape_latex` sanitization function on all dynamic string fields before insertion into the template. The escaping function must securely replace backslashes (using a temporary non-conflicting placeholder, e.g., `BACKSLASHPLACEHOLDER`, replacing backslashes, then `_`, etc., and finally resolving the placeholder to `\textbackslash{}`) to prevent double-escaping and effectively neutralize format string injection.
