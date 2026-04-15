## 2024-03-22 - [FastAPI Path Parameter URL-Decoding Bug]
**Vulnerability:** Path traversal in a catch-all route that serves static files using FastAPI path parameters.
**Learning:** FastAPI's '{path:path}' definition decodes URL-encoded payloads like '%2e%2e%2f' after Starlette's initial sanitization, bypassing standard sanitizations and directly feeding dangerous input to 'pathlib.Path'.
**Prevention:** Always verify paths mathematically using 'candidate.resolve().relative_to(base_dir.resolve())' instead of trusting the input when generating file paths.

## 2024-05-18 - Fix LaTeX injection in PDF report generation
**Vulnerability:** User-provided patient details, dictionary keys, and diagnostic values were concatenated into a LaTeX string without escaping. This creates a critical server-side injection vulnerability where arbitrary LaTeX code can be executed (e.g. `\input{/etc/passwd}` or external commands).
**Learning:** Python f-strings used for rendering LaTeX templates must safely escape all user inputs. In addition to obvious scalar fields, dynamic data like dictionary keys generated via API payloads must also be escaped.
**Prevention:** Implement an explicit single-pass LaTeX escape function that converts all special characters (such as `\`, `&`, `%`, `$`, `#`, `_`, `{`, `}`, `~`, `^`) into their text equivalents, taking care to escape backslashes first, before interpolating inputs into `.tex` templates.
