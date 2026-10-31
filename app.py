"""
Flask web app: upload resume PDFs + a job description -> ranked match scores
-> a human reviewer approves/rejects every candidate (human-in-the-loop).

Run:  python app.py      then open http://127.0.0.1:5000

The recommendation comes from the transparent JD-match score. The trained ML
model is shown only in an "experimental" panel because it has no real predictive
signal on dataset.csv (see README). It never influences the recommendation.
"""
import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from flask import (Flask, abort, flash, g, redirect, render_template, request,
                   session, url_for)

import resume_parser as rp

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "screening.db"
MODEL_PATH = BASE / "model" / "screening_model.pkl"
META_PATH = BASE / "model" / "model_meta.json"
SECRET_KEY_PATH = BASE / ".secret_key"


def load_secret_key() -> str:
    configured = os.environ.get("SECRET_KEY")
    if configured:
        return configured
    if SECRET_KEY_PATH.exists():
        return SECRET_KEY_PATH.read_text().strip()
    secret = secrets.token_hex(32)
    SECRET_KEY_PATH.write_text(secret)
    return secret


app = Flask(__name__)
app.config["SECRET_KEY"] = load_secret_key()
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024   # 25 MB per request
MAX_FILES = 20

MODEL = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
META = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
DEFAULT_ROLES = [
    "Business_Analyst",
    "Cybersecurity_Analyst",
    "Data_Analyst",
    "Data_Scientist",
    "DevOps_Engineer",
    "HR_Executive",
    "Project_Manager",
    "QA_Engineer",
    "Software_Engineer",
    "UI_UX_Designer",
]
ROLES = META.get("job_roles") or DEFAULT_ROLES

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch TEXT, filename TEXT, job_role TEXT,
  match_score REAL, recommendation TEXT,
  years_exp INTEGER, skills_matched INTEGER, matched_skills TEXT, missing_skills TEXT,
  ml_probability REAL,
  decision TEXT DEFAULT 'Pending', reviewer_note TEXT,
  created_at TEXT, decided_at TEXT
);"""


def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute(SCHEMA)
    return g.db


@app.teardown_appcontext
def close_db(_):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def csrf_token() -> str:
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]


app.jinja_env.globals["csrf_token"] = csrf_token


def check_csrf():
    if not secrets.compare_digest(request.form.get("csrf", ""), session.get("csrf", "x")):
        abort(400, "Invalid form token, reload the page and try again.")


def ml_probability(role: str, a: dict):
    if MODEL is None:
        return None
    row = pd.DataFrame([{"job_role": role, "years_exp": a["years_exp"],
                         "skills_matched": a["skills_matched"],
                         "education_score": a["education_score"],
                         "keyword_density": a["keyword_density"]}])
    return float(MODEL.predict_proba(row)[0, 1])


@app.get("/")
def index():
    return render_template("index.html", roles=ROLES)


@app.post("/screen")
def screen():
    check_csrf()
    jd = request.form.get("jd", "").strip()
    role = request.form.get("job_role", "")
    files = [f for f in request.files.getlist("resumes") if f and f.filename]
    if role not in ROLES:
        flash("Pick a valid job role.")
    elif len(jd) < 30:
        flash("Paste a job description (at least 30 characters).")
    elif not files:
        flash("Upload at least one resume PDF.")
    elif len(files) > MAX_FILES:
        flash(f"Upload at most {MAX_FILES} resumes at a time.")
    else:
        batch = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        results, errors = [], []
        for f in files:
            data = f.read()
            if not f.filename.lower().endswith(".pdf") or not data.startswith(b"%PDF"):
                errors.append(f"{f.filename}: not a valid PDF")
                continue
            try:
                text = rp.pdf_to_text(data)
            except Exception:
                errors.append(f"{f.filename}: could not be read")
                continue
            if len(text.strip()) < 50:
                errors.append(f"{f.filename}: no extractable text (scanned image PDF?)")
                continue
            a = rp.analyse(text, jd)          # resume text is NOT stored
            prob = ml_probability(role, a)
            cur = db().execute(
                "INSERT INTO candidates(batch,filename,job_role,match_score,recommendation,"
                "years_exp,skills_matched,matched_skills,missing_skills,ml_probability,created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (batch, Path(f.filename).name, role, a["match_score"], a["recommendation"],
                 a["years_exp"], a["skills_matched"], ", ".join(a["matched_skills"]),
                 ", ".join(a["missing_skills"]), prob,
                 datetime.now(timezone.utc).isoformat(timespec="seconds")))
            results.append({"id": cur.lastrowid, "filename": Path(f.filename).name,
                            "decision": "Pending", "ml_probability": prob, **a})
        db().commit()
        results.sort(key=lambda r: r["match_score"], reverse=True)
        return render_template("results.html", results=results, errors=errors, role=role,
                               meta=META, cfg=rp)
    return redirect(url_for("index"))


@app.post("/decision/<int:cid>")
def decision(cid):
    check_csrf()
    choice = request.form.get("decision")
    if choice not in ("Approved", "Rejected"):
        abort(400)
    note = request.form.get("note", "").strip()[:300]
    cur = db().execute("UPDATE candidates SET decision=?, reviewer_note=?, decided_at=? WHERE id=?",
                       (choice, note, datetime.now(timezone.utc).isoformat(timespec="seconds"), cid))
    db().commit()
    if cur.rowcount == 0:
        abort(404)
    return redirect(url_for("review_log"))


@app.get("/review")
def review_log():
    rows = db().execute("SELECT * FROM candidates ORDER BY id DESC LIMIT 200").fetchall()
    counts = {r["decision"]: r["n"] for r in db().execute(
        "SELECT decision, COUNT(*) n FROM candidates GROUP BY decision")}
    return render_template("review.html", rows=rows, counts=counts)


@app.get("/about")
def about():
    return render_template("about.html", meta=META, cfg=rp)


if __name__ == "__main__":
    app.run(debug=False)
