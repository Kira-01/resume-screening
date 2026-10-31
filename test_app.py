"""Smoke tests: python test_app.py"""
import io, re
from pathlib import Path
import app as A
import resume_parser as rp

S = Path(__file__).parent / "samples"
jd = (S / "sample_job_description.txt").read_text()

def tok(c):
    html = c.get("/").get_data(as_text=True)
    return re.search(r'name="csrf" value="([^"]+)"', html).group(1)

def test_parser():
    a = rp.analyse(rp.pdf_to_text((S/"resume_strong_match.pdf").read_bytes()), jd)
    b = rp.analyse(rp.pdf_to_text((S/"resume_weak_match.pdf").read_bytes()), jd)
    assert a["match_score"] > b["match_score"] and a["recommendation"] == "Recommend shortlist", a
    assert a["years_exp"] == 6 and a["education_score"] == 88, a
    assert "python" in a["matched_skills"] and b["skills_matched"] <= 1

def test_flow():
    c = A.app.test_client()
    t = tok(c)
    files = [(io.BytesIO((S/n).read_bytes()), n) for n in
             ["resume_strong_match.pdf", "resume_partial_match.pdf", "resume_weak_match.pdf"]]
    files.append((io.BytesIO((S/"not_a_resume.pdf").read_bytes()), "not_a_resume.pdf"))
    r = c.post("/screen", data={"csrf": t, "job_role": "Data_Analyst", "jd": jd,
                                "resumes": files}, content_type="multipart/form-data")
    html = r.get_data(as_text=True)
    assert r.status_code == 200 and "not a valid PDF" in html
    assert html.index("resume_strong_match") < html.index("resume_weak_match")
    cid = int(re.search(r"/decision/(\d+)", html).group(1))
    r = c.post(f"/decision/{cid}", data={"csrf": t, "decision": "Approved", "note": "ok"})
    assert r.status_code == 302 and r.location.endswith("/review")
    assert "Approved" in c.get("/review").get_data(as_text=True)

def test_all_roles_are_available():
    html = A.app.test_client().get("/").get_data(as_text=True)
    assert len(A.ROLES) == 10
    assert all(f'value="{role}"' in html for role in A.ROLES)

def test_pending_reviews_have_decision_buttons():
    c = A.app.test_client()
    t = tok(c)
    with A.app.app_context():
        A.db().execute(
            "INSERT INTO candidates(batch,filename,job_role,match_score,recommendation,created_at)"
            " VALUES(?,?,?,?,?,?)",
            ("test", "pending.pdf", "Data_Analyst", 50, "Review", "2026-01-01T00:00:00+00:00"))
        A.db().commit()
    html = c.get("/review").get_data(as_text=True)
    assert 'value="Approved"' in html and 'value="Rejected"' in html

def test_security_and_validation():
    c = A.app.test_client(); tok(c)
    assert c.post("/decision/1", data={"decision": "Approved"}).status_code == 400   # no CSRF
    t = tok(c)
    assert c.post("/decision/1", data={"csrf": t, "decision": "Hacked"}).status_code == 400
    r = c.post("/screen", data={"csrf": t, "job_role": "Nope", "jd": "x"}, follow_redirects=True)
    assert "valid job role" in r.get_data(as_text=True)
    assert c.get("/about").status_code == 200

def test_nlp_engine_reported():
    html = A.app.test_client().get("/about").get_data(as_text=True)
    assert "NLP engine" in html
    if rp.SPACY_ACTIVE:   # only runs the spaCy path where spaCy + model are installed
        assert "spaCy (en_core_web_sm)" in html
        a = rp.analyse(rp.pdf_to_text((S/"resume_strong_match.pdf").read_bytes()), jd)
        assert a["recommendation"] == "Recommend shortlist", a

if __name__ == "__main__":
    for f in (test_parser, test_flow, test_all_roles_are_available,
              test_pending_reviews_have_decision_buttons,
              test_security_and_validation, test_nlp_engine_reported):
        f(); print("PASS", f.__name__)
