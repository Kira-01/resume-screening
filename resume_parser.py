"""
Resume parsing and job-description matching.

Pipeline: PDF bytes -> text (pdfminer) -> skills / years / education extraction
-> TF-IDF cosine similarity with the JD -> transparent match score.

spaCy is used for lemmatisation when `en_core_web_sm` is installed; otherwise a
plain regex tokenizer is used. Results are slightly better with spaCy but the
system works without it.

Bias control: the candidate's name is never extracted or used in scoring.
"""
import re
from io import BytesIO

from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:  # optional
    import spacy
    _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except Exception:  # spaCy or model not installed
    _NLP = None
SPACY_ACTIVE = _NLP is not None

# Transparent, editable scoring configuration -------------------------------
W_SKILLS = 0.60          # weight of JD-skill coverage in the match score
W_TFIDF = 0.40           # weight of TF-IDF text similarity
SHORTLIST_AT = 60.0      # score >= this  -> "Recommend shortlist"
REJECT_BELOW = 35.0      # score <  this  -> "Recommend reject", else "Borderline"

SKILL_VOCAB = [
    # programming / data
    "python", "java", "javascript", "typescript", "c++", "sql", "mysql", "postgresql", "mongodb",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "machine learning",
    "deep learning", "nlp", "data analysis", "data visualization", "statistics", "excel",
    "power bi", "tableau", "etl", "data cleaning", "r",
    # software engineering
    "spring", "spring boot", "rest api", "flask", "django", "react", "node.js", "html", "css",
    "git", "docker", "kubernetes", "aws", "azure", "ci/cd", "microservices", "data structures",
    "algorithms", "object oriented", "linux", "agile",
    # QA
    "selenium", "junit", "pytest", "test automation", "manual testing", "test cases",
    "jira", "postman", "api testing", "regression testing", "performance testing", "bug tracking",
    # project management
    "project management", "scrum", "kanban", "stakeholder management", "risk management",
    "budgeting", "ms project", "pmp", "roadmap", "sprint planning", "resource planning",
    # HR
    "recruitment", "talent acquisition", "onboarding", "payroll", "employee engagement",
    "performance management", "hris", "labour law", "sourcing", "interviewing",
    "compensation", "training and development", "communication", "leadership",
    # data science / DevOps / BA / UX / security
    "feature engineering", "model deployment", "mlops", "a/b testing", "jenkins", "terraform",
    "ansible", "prometheus", "grafana", "bash", "requirements gathering", "user stories",
    "process mapping", "business analysis", "figma", "wireframing", "prototyping",
    "user research", "usability testing", "design systems", "siem", "penetration testing",
    "vulnerability assessment", "owasp", "incident response", "network security", "iso 27001",
    "splunk", "firewall",
]

_DEGREES = [  # (regex, score) -> heuristic education score on the dataset's 60-100 scale
    (r"\b(ph\.?d|doctorate)\b", 95),
    (r"\b(m\.?tech|m\.?sc|m\.?e\b|mca|mba|master)", 88),
    (r"\b(b\.?tech|b\.?e\b|b\.?sc|bca|bachelor)", 78),
    (r"\bdiploma\b", 65),
]


def _skill_pattern(skill: str) -> re.Pattern:
    return re.compile(r"(?<![\w+#])" + re.escape(skill) + r"(?![\w+#])", re.I)


_SKILL_RE = {s: _skill_pattern(s) for s in SKILL_VOCAB}


def pdf_to_text(data: bytes) -> str:
    return extract_text(BytesIO(data)) or ""


def find_skills(text: str) -> set:
    return {s for s, rx in _SKILL_RE.items() if rx.search(text)}


def estimate_years(text: str) -> int:
    """Largest explicit '<N> years' claim, else sum of year ranges (e.g. 2019 - 2022)."""
    claims = [int(m) for m in re.findall(r"(\d{1,2})\s*\+?\s*(?:years|yrs)", text, flags=re.I)]
    claims = [c for c in claims if c <= 40]
    if claims:
        return max(claims)
    total = 0
    for a, b in re.findall(r"((?:19|20)\d{2})\s*[-\u2013to ]+\s*((?:19|20)\d{2}|present|current)",
                           text, flags=re.I):
        end = 2026 if not b.isdigit() else int(b)
        if int(a) <= end:
            total += end - int(a)
    return min(total, 40)


def education_score(text: str) -> int:
    for rx, score in _DEGREES:
        if re.search(rx, text, flags=re.I):
            return score
    return 60


def _prep(text: str) -> str:
    text = re.sub(r"[^a-z0-9+#./ ]+", " ", text.lower())
    if _NLP is not None:
        return " ".join(t.lemma_ for t in _NLP(text) if not t.is_space)
    return re.sub(r"\s+", " ", text)


def tfidf_similarity(jd: str, resume: str) -> float:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    try:
        m = vec.fit_transform([_prep(jd), _prep(resume)])
    except ValueError:  # empty vocabulary
        return 0.0
    return float(cosine_similarity(m[0], m[1])[0, 0])


def analyse(resume_text: str, jd_text: str) -> dict:
    jd_skills = find_skills(jd_text)
    res_skills = find_skills(resume_text)
    matched = sorted(jd_skills & res_skills)
    missing = sorted(jd_skills - res_skills)
    coverage = len(matched) / len(jd_skills) if jd_skills else 0.0
    sim = tfidf_similarity(jd_text, resume_text)
    score = 100 * (W_SKILLS * coverage + W_TFIDF * sim)

    jd_years = [int(m) for m in re.findall(r"(\d{1,2})\s*\+?\s*(?:years|yrs)", jd_text, flags=re.I)]
    years = estimate_years(resume_text)

    if not jd_skills:
        rec = "Cannot score: no recognised skills in the job description"
    elif score >= SHORTLIST_AT:
        rec = "Recommend shortlist"
    elif score < REJECT_BELOW:
        rec = "Recommend reject"
    else:
        rec = "Borderline"

    return {
        "match_score": round(score, 1),
        "skill_coverage": round(coverage, 3),
        "tfidf_similarity": round(sim, 3),
        "matched_skills": matched,
        "missing_skills": missing,
        "jd_skill_count": len(jd_skills),
        "years_exp": years,
        "jd_required_years": max(jd_years) if jd_years else None,
        "education_score": education_score(resume_text),
        # features in the dataset's definition, for the experimental ML panel:
        "skills_matched": len(matched),
        "keyword_density": round(coverage, 3),
        "recommendation": rec,
        "text_chars": len(resume_text),
    }
