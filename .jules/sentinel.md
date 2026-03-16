## 2024-05-18 - Fix LaTeX injection vulnerability in PDF report generation
**Vulnerability:** Unescaped user input passed directly to pdflatex during PDF report generation (LaTeX Injection).
**Learning:** Even if pdflatex isn't run with -shell-escape, unescaped input allows arbitrary file reads (via \input) and can crash the LaTeX compiler, causing DoS. Using a single-pass regex replacement logic to avoid sequential replacement bugs.
**Prevention:** Always escape LaTeX special characters (%, $, &, #, _, {, }, ~, ^, \) using a single-pass regex substitution before interpolating data into a .tex template.
