import subprocess
import tempfile
import shutil
import cv2
import re
from pathlib import Path
from report_generator import ReportRequest, SHORT_NAMES, RISK_LEVEL

def tex_escape(text):
    text = str(text)
    conv = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"
    }
    return re.sub(r'[&%$#_{}~^\\]', lambda m: conv[m.group()], text)

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
            
        sn_escaped = tex_escape(SHORT_NAMES.get(req.ai_pred_key, req.ai_pred_key))
        risk_escaped = tex_escape(RISK_LEVEL.get(req.ai_pred_key, "UNKNOWN"))
        
        probs = "\n".join([f"\\item \\textbf{{{tex_escape(SHORT_NAMES.get(k, k))}}}: {v*100:.1f}\\%" for k, v in req.probabilities.items()])
        
        date_escaped = tex_escape(req.server_timestamp)
        session_escaped = tex_escape(req.session_id[:16])
        name_escaped = tex_escape(req.patient.patient_name)
        id_escaped = tex_escape(req.patient.patient_id)
        dob_escaped = tex_escape(req.patient.date_of_birth)
        gender_escaped = tex_escape(req.patient.gender)

        tex = f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{helvet}}
\\renewcommand{{\\familydefault}}{{\\sfdefault}}
\\begin{{document}}
\\begin{{center}}
    {{\\LARGE \\textbf{{TECNOMATE CLINICAL AI - DIAGNOSTIC REPORT}}}} \\\\[0.5cm]
    \\textbf{{Date:}} {date_escaped} \\quad \\textbf{{Session:}} {session_escaped}
\\end{{center}}
\\hrule \\vspace{{0.5cm}}
\\textbf{{Patient Name:}} {name_escaped} \\\\
\\textbf{{Patient ID:}} {id_escaped} \\\\
\\textbf{{DOB:}} {dob_escaped} \\quad \\textbf{{Gender:}} {gender_escaped}
\\vspace{{0.5cm}} \\hrule \\vspace{{0.5cm}}
\\begin{{center}}
{scan_img} \\quad {cam_img}
\\end{{center}}
\\vspace{{0.5cm}} \\hrule \\vspace{{0.5cm}}
\\textbf{{Prediction:}} {sn_escaped} \\\\
\\textbf{{Confidence:}} {req.ai_confidence*100:.1f}\\% \\quad \\textbf{{Risk:}} {risk_escaped}
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
