import unittest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path
from latex_report import build_latex_report
from report_generator import ReportRequest, PatientInfo, UncertaintyInfo

class TestLatexReport(unittest.TestCase):
    @patch('subprocess.run')
    def test_latex_escaping(self, mock_subprocess):
        # Setup mock for subprocess.run
        mock_subprocess.return_value = MagicMock(returncode=0)

        # We need to monkeypatch Path specifically for the test to intercept write_text
        original_path = Path
        class MockPath(original_path):
            last_tex_content = None

            def write_text(self, content, *args, **kwargs):
                if self.name == 'r.tex':
                    MockPath.last_tex_content = content
                super().write_text(content, *args, **kwargs)

            def read_bytes(self, *args, **kwargs):
                if self.name == 'r.pdf':
                    return b"fake pdf content"
                return super().read_bytes(*args, **kwargs)

        with patch('latex_report.Path', MockPath):
            # Create a request with malicious LaTeX injection
            req = ReportRequest(
                session_id="test_session_12345",
                filename="test.jpg",
                scan_type="Brain MRI",
                ai_pred_key="glioma",
                ai_confidence=0.95,
                probabilities={"glioma": 0.95, "notumor": 0.05},
                doctor_choice_key="glioma",
                patient=PatientInfo(
                    patient_name=r"\input{/etc/passwd}",
                    patient_id=r"ID_123&456",
                    date_of_birth=r"01/01/1990_%",
                    gender=r"M $ ^ ~",
                ),
                server_timestamp=r"2023-10-27_10:00:00"
            )

            # This should not raise an exception and should call pdflatex
            try:
                pdf_bytes = build_latex_report(req)
                self.assertEqual(pdf_bytes, b"fake pdf content")
            except Exception as e:
                if "pdflatex not found" in str(e):
                    # In test environment, pdflatex might not be installed,
                    # but we only care about testing the escaping logic before that step.
                    pass
                else:
                    raise

            # Assert that the malicious inputs were correctly escaped
            tex_content = MockPath.last_tex_content
            self.assertIn(r"\textbackslash{}input\{/etc/passwd\}", tex_content)
            self.assertIn(r"ID\_123\&456", tex_content)
            self.assertIn(r"01/01/1990\_\%", tex_content)
            self.assertIn(r"M \$ \textasciicircum{} \textasciitilde{}", tex_content)
            self.assertIn(r"2023-10-27\_10:00:00", tex_content)

if __name__ == '__main__':
    unittest.main()
