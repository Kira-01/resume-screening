"""Creates fictional sample resumes for testing. No real people."""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUT = Path(__file__).parent
RESUMES = {
 "resume_strong_match.pdf": ["Candidate A", "Data Analyst, 6 years experience",
  "Skills: Python, SQL, pandas, numpy, Power BI, Tableau, Excel, statistics, data visualization, data cleaning, ETL, machine learning",
  "Experience: 2019 - 2025 Senior Data Analyst. Built dashboards and automated ETL pipelines.",
  "Education: M.Tech in Computer Science"],
 "resume_partial_match.pdf": ["Candidate B", "Junior analyst, 2 years experience",
  "Skills: Excel, SQL, communication",
  "Experience: 2023 - 2025 Reporting Assistant preparing weekly Excel reports.",
  "Education: B.Sc Mathematics"],
 "resume_weak_match.pdf": ["Candidate C", "HR coordinator, 4 years experience",
  "Skills: recruitment, onboarding, payroll, employee engagement",
  "Experience: 2021 - 2025 HR Coordinator handling onboarding and payroll.",
  "Education: MBA"],
}
for name, lines in RESUMES.items():
    c = canvas.Canvas(str(OUT / name), pagesize=A4)
    y = 800
    for ln in lines:
        c.drawString(50, y, ln[:110]); y -= 24
    c.save()
(OUT / "sample_job_description.txt").write_text(
 "Data Analyst. We need 3+ years of experience. Required skills: Python, SQL, pandas, "
 "Power BI or Tableau, data visualization, statistics, Excel, ETL and data cleaning. "
 "Machine learning knowledge is a plus. Strong communication skills.")
(OUT / "not_a_resume.pdf").write_bytes(b"this is not a pdf")
print("samples created")
