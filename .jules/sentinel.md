## 2024-05-24 - LaTeX Injection in PDF Report Generation
**Vulnerability:** The backend `latex_report.py` directly interpolated unescaped user inputs (like patient names and IDs) into a LaTeX template string, which was then compiled via a shell call to `pdflatex`. This allowed arbitrary LaTeX command execution.
**Learning:** Even if data is only used for PDF generation, template engines (like LaTeX) that allow shell-escapes or complex macro logic can be a vector for severe injection attacks.
**Prevention:** Always use a single-pass regex replacement function to correctly escape LaTeX special characters (`\`, `&`, `%`, `$`, `#`, `_`, `{`, `}`, `~`, `^`) before injecting any user-controlled string into a `.tex` template.
