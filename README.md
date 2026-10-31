# AI-Based Resume Screening and Candidate Shortlisting System
HCLTech Career Shaper Programme, Project 34 (Digital-Data-AI / AIML)

## Quick start
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm     # NLP model for spaCy (app falls back to regex without it)
python app.py              # web app at http://127.0.0.1:5000
```

## What is in the folder
| Path | Purpose |
|---|---|
| `resume_parser.py` | PDF text extraction, skill/years/education extraction, TF-IDF + skill-coverage match score |
| `app.py`, `templates/` | Flask app with human-in-the-loop approve/reject and SQLite review log |
| `model/` | Saved sklearn Pipeline |
| `samples/`, `test_app.py` | Fictional sample resumes and smoke tests |

## NLP: where spaCy and TF-IDF are used
* **TF-IDF** (scikit-learn, unigrams + bigrams, sublinear tf): used in the web app to compare each resume's text with the job description (cosine similarity, 40% of the match score). It is **not** used in the ML model, because `dataset.csv` contains numeric features only (no resume text) to vectorise.
* **spaCy** (`en_core_web_sm`): used to lemmatise resume and JD text before TF-IDF when installed, and the About page shows which engine is active. Skill extraction uses a deterministic regex vocabulary, so results do not depend on spaCy.
* If spaCy is not installed the app falls back to plain regex tokenisation and still works. The spaCy path could not be run in the build environment (no network), so run `python test_app.py` on your machine after installing spaCy; the test exercises that path automatically.

## Results
* 5,000 rows, no nulls or duplicates; 34.6% shortlisted.
* `predicted_score` is excluded: it is an AI-generated score of the same candidates, so using it would be circular. Adding it changed AUC by 0.000.
* 5-fold CV ROC-AUC: Logistic Regression 0.735, Random Forest 0.732, Gradient Boosting 0.731, SVM 0.726 (majority-class baseline 0.50). Accuracy 0.675 vs baseline 0.650, so report ROC-AUC and F1 rather than accuracy alone.

## How the app decides
The recommendation comes from a transparent score (60% JD-skill coverage + 40% TF-IDF similarity; thresholds 60/35), **not** from the ML model. The ML probability is displayed as "experimental" only. Weights, thresholds and the skill vocabulary are design choices, editable at the top of `resume_parser.py`.

Mapping a resume to the dataset's features (heuristic): `skills_matched` = JD skills found in the resume; `keyword_density` = skills matched / JD skills; `years_exp` = largest "N years" claim or sum of date ranges; `education_score` = degree level (PhD 95, Master 88, Bachelor 78, Diploma 65, other 60).

## Limitations
* Scanned PDFs are rejected (no OCR). The skill vocabulary is fixed (about 130 terms).
* File names are stored in the review log and may contain candidate names; resume text is never stored. Use an access-controlled machine and add consent handling before real use (DPDP Act 2023).
* Not for production: no login, single-process SQLite.

## Ethics
Decision support only; a human decides every case. Not the sole selection criterion.
