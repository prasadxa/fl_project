import subprocess
import tempfile
import shutil
import cv2
from pathlib import Path
from report_generator import ReportRequest, SHORT_NAMES, RISK_LEVEL

def _escape_latex(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    latex_special_chars = {
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
    return s.translate(str.maketrans(latex_special_chars))

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
            
        safe_sn = _escape_latex(SHORT_NAMES.get(req.ai_pred_key, req.ai_pred_key))
        safe_risk = _escape_latex(RISK_LEVEL.get(req.ai_pred_key, "UNKNOWN"))
        
        probs = "\n".join([f"\\item \\textbf{{{_escape_latex(SHORT_NAMES.get(k, k))}}}: {v*100:.1f}\\%" for k, v in req.probabilities.items()])
        
        safe_timestamp = _escape_latex(req.server_timestamp)
        safe_session = _escape_latex(req.session_id[:16])
        safe_p_name = _escape_latex(req.patient.patient_name)
        safe_p_id = _escape_latex(req.patient.patient_id)
        safe_p_dob = _escape_latex(req.patient.date_of_birth)
        safe_p_gender = _escape_latex(req.patient.gender)

        tex = f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{helvet}}
\\renewcommand{{\\familydefault}}{{\\sfdefault}}
\\begin{{document}}
\\begin{{center}}
    {{\\LARGE \\textbf{{TECNOMATE CLINICAL AI - DIAGNOSTIC REPORT}}}} \\\\[0.5cm]
    \\textbf{{Date:}} {safe_timestamp} \\quad \\textbf{{Session:}} {safe_session}
\\end{{center}}
\\hrule \\vspace{{0.5cm}}
\\textbf{{Patient Name:}} {safe_p_name} \\\\
\\textbf{{Patient ID:}} {safe_p_id} \\\\
\\textbf{{DOB:}} {safe_p_dob} \\quad \\textbf{{Gender:}} {safe_p_gender}
\\vspace{{0.5cm}} \\hrule \\vspace{{0.5cm}}
\\begin{{center}}
{scan_img} \\quad {cam_img}
\\end{{center}}
\\vspace{{0.5cm}} \\hrule \\vspace{{0.5cm}}
\\textbf{{Prediction:}} {safe_sn} \\\\
\\textbf{{Confidence:}} {req.ai_confidence*100:.1f}\\% \\quad \\textbf{{Risk:}} {safe_risk}
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
