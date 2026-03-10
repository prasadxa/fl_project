import subprocess
import tempfile
import shutil
import cv2
from pathlib import Path
from report_generator import ReportRequest, SHORT_NAMES, RISK_LEVEL

def escape_latex(s: str) -> str:
    """Escapes strings for LaTeX to prevent injection."""
    if not s:
        return ""
    s = str(s)
    special_chars = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}',
    }
    return "".join(special_chars.get(c, c) for c in s)

def build_latex_report(req: ReportRequest) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        
        scan_img, cam_img = "", ""
        if req.scan_image_path and Path(req.scan_image_path).exists():
            shutil.copy(req.scan_image_path, p / "scan.jpg")
            scan_img = r"\includegraphics[width=0.45\textwidth]{scan.jpg}"
            
        if req.gradcam_image is not None:
            cv2.imwrite(str(p / "cam.jpg"), cv2.cvtColor(req.gradcam_image, cv2.COLOR_RGB2BGR))
            cam_img = r"\includegraphics[width=0.45\textwidth]{cam.jpg}"
            
        sn = SHORT_NAMES.get(req.ai_pred_key, req.ai_pred_key)
        risk = RISK_LEVEL.get(req.ai_pred_key, "UNKNOWN")
        
        probs = "\n".join([f"\\item \\textbf{{{SHORT_NAMES.get(k, k)}}}: {v*100:.1f}\\%" for k, v in req.probabilities.items()])
        
        safe_session_id = escape_latex(req.session_id[:16])
        safe_patient_name = escape_latex(req.patient.patient_name)
        safe_patient_id = escape_latex(req.patient.patient_id)
        safe_dob = escape_latex(req.patient.date_of_birth)
        safe_gender = escape_latex(req.patient.gender)

        tex = f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{helvet}}
\\renewcommand{{\\familydefault}}{{\\sfdefault}}
\\begin{{document}}
\\begin{{center}}
    {{\\LARGE \\textbf{{TECNOMATE CLINICAL AI - DIAGNOSTIC REPORT}}}} \\\\[0.5cm]
    \\textbf{{Date:}} {req.server_timestamp} \\quad \\textbf{{Session:}} {safe_session_id}
\\end{{center}}
\\hrule \\vspace{{0.5cm}}
\\textbf{{Patient Name:}} {safe_patient_name} \\\\
\\textbf{{Patient ID:}} {safe_patient_id} \\\\
\\textbf{{DOB:}} {safe_dob} \\quad \\textbf{{Gender:}} {safe_gender}
\\vspace{{0.5cm}} \\hrule \\vspace{{0.5cm}}
\\begin{{center}}
{scan_img} \\quad {cam_img}
\\end{{center}}
\\vspace{{0.5cm}} \\hrule \\vspace{{0.5cm}}
\\textbf{{Prediction:}} {sn} \\\\
\\textbf{{Confidence:}} {req.ai_confidence*100:.1f}\\% \\quad \\textbf{{Risk:}} {risk}
\\begin{{itemize}}
{probs}
\\end{{itemize}}
\\vspace{{1cm}}
\\textbf{{Disclaimer:}} AI-generated report. Requires clinician review.
\\end{{document}}"""
        
        (p / "r.tex").write_text(tex)
        
        try:
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "r.tex"], cwd=p, check=True, capture_output=True)
        except FileNotFoundError:
            raise Exception("pdflatex not found.")
        except subprocess.CalledProcessError as e:
            raise Exception(f"LaTeX failed: {e.stderr.decode()}")
            
        return (p / "r.pdf").read_bytes()
