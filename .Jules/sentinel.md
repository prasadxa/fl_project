
## $(date +%Y-%m-%d) - [LaTeX Injection in PDF Generator]
**Vulnerability:** User-controlled inputs (`patient_name`, `session_id`, etc.) were directly interpolated into a LaTeX template (`backend/latex_report.py`) using f-strings without escaping. Because the template is compiled by `pdflatex` via `subprocess.run`, an attacker could inject arbitrary LaTeX commands (e.g., `\input{/etc/passwd}`).
**Learning:** Even when shell execution isn't enabled (e.g., `--shell-escape`), `pdflatex` still natively supports reading arbitrary files on the local filesystem via `\input{}` or `\read`. Any user input entering a LaTeX template must be aggressively escaped.
**Prevention:** Implement a robust, single-pass escaping function (`_escape_latex`) to replace LaTeX special characters (`\`, `{`, `}`, `#`, `$`, `%`, `&`, `_`, `^`, `~`) with their literal string equivalents (`\textbackslash{}`, `\{`, etc.) before interpolation.
