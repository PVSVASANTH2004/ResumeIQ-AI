import re

import numpy as np

from core.models import (Engine1Output, Engine2Output, Engine3Output,
                         Engine4Output, Engine5Output, ScoreBreakdown)
from engines.candidate_quality.learning_progression import progression_to_score
from engines.explainability.jd_intent_detector import detect_jd_intent

_SECTION_KWS = ["experience", "education", "skills", "projects", "summary", "certifications"]
_DATE_RE  = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,]*\d{4}|\b\d{4}\b", re.I
)
_EMAIL_RE  = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_BULLET_RE = re.compile(r"[•\-–●▪\*]\s+\S")

# ── Education scoring ─────────────────────────────────────────────────────────
_DEGREE_LEVEL = {
    "phd":      re.compile(r"\b(phd|ph\.d|doctorate|doctor)\b", re.I),
    "master":   re.compile(r"\b(master|m\.s\.|m\.tech|mba|m\.e\.|msc)\b", re.I),
    "bachelor": re.compile(r"\b(bachelor|b\.tech|b\.e\.|b\.s\.|b\.sc|undergraduate|engineering)\b", re.I),
}
_FIELD_TERMS = [
    "computer science", "information technology", "software engineering",
    "electrical engineering", "electronics", "mathematics", "physics",
    "data science", "artificial intelligence", "machine learning",
]
_GPA_RE = re.compile(r"gpa\s*[>≥:]\s*(\d+\.?\d*)", re.I)


def _education_score(e1: Engine1Output) -> float:
    """
    Compare resume education against JD education requirements.
    Checks: degree level, field of study, GPA threshold.
    Returns 0–1.
    """
    jd_text   = e1.parsed_jd.raw_text
    resume_edu = e1.parsed_resume.education

    if not resume_edu or not jd_text:
        return 0.0

    best     = resume_edu[0]
    jd_lower = jd_text.lower()
    score    = 0.0
    checks   = 0

    # 1. Degree level
    required_level = None
    for level, pattern in _DEGREE_LEVEL.items():
        if pattern.search(jd_text):
            required_level = level
            break

    if required_level:
        checks += 1
        candidate = (best.degree + " " + best.field).lower()
        if _DEGREE_LEVEL[required_level].search(candidate):
            score += 1.0
        elif required_level == "bachelor" and re.search(
            r"\b(b\.tech|b\.e\.|bachelor|engineering)\b", candidate, re.I
        ):
            score += 1.0  # B.Tech is a bachelor's degree

    # 2. Field of study
    jd_fields = [f for f in _FIELD_TERMS if f in jd_lower]
    if jd_fields:
        checks += 1
        candidate_field = (best.field + " " + best.degree).lower()
        if any(f in candidate_field for f in jd_fields):
            score += 1.0
        elif "computer science" in jd_lower and "engineering" in candidate_field:
            score += 0.9  # CSE ≈ CS + Engineering

    # 3. GPA requirement
    gpa_match = _GPA_RE.search(jd_text)
    if gpa_match:
        checks += 1
        required_gpa   = float(gpa_match.group(1))
        candidate_gpa  = best.gpa or 0.0
        if candidate_gpa > 0:
            if candidate_gpa >= required_gpa:
                score += 1.0
            elif candidate_gpa >= required_gpa - 0.3:
                score += 0.7  # close miss

    if checks == 0:
        return 0.80 if resume_edu else 0.0

    return round(min(score / checks, 1.0), 4)


def _readability_score(text: str) -> float:
    """
    Resume-specific readability heuristic (0–1).
    Flesch Reading Ease is unsuitable for resumes — technical jargon,
    bullet fragments, and absent punctuation yield near-zero scores
    regardless of actual resume quality.
    """
    score = 0.0

    sections_found = sum(
        1 for kw in _SECTION_KWS if re.search(r"\b" + kw + r"\b", text, re.I)
    )
    score += min(sections_found / 4.0, 1.0) * 0.30

    bullet_count = len(_BULLET_RE.findall(text))
    score += min(bullet_count / 10.0, 1.0) * 0.25

    word_count = len(text.split())
    if 200 <= word_count <= 1000:
        score += 0.20
    elif word_count < 200:
        score += (word_count / 200.0) * 0.20
    else:
        score += max(0.0, (2000 - word_count) / 1000.0) * 0.20

    date_count = len(_DATE_RE.findall(text))
    score += min(date_count / 4.0, 1.0) * 0.15

    score += 0.10 if _EMAIL_RE.search(text) else 0.0

    return round(min(score, 1.0), 4)


def compose(
    e1: Engine1Output,
    e2: Engine2Output,
    e3: Engine3Output,
    e4: Engine4Output,
    e5: Engine5Output,
) -> ScoreBreakdown:
    # ── Dynamic weights based on JD intent ───────────────────────────────────
    weights = detect_jd_intent(e1.parsed_jd.raw_text, e1.parsed_jd)

    # ── Component scores ─────────────────────────────────────────────────────
    jd_skill_set  = {s.lower() for s in (
        e1.parsed_jd.required_skills + e1.parsed_jd.preferred_skills
    )}
    jd_evidence   = {s: v for s, v in e3.evidence_scores.items()
                     if s.lower() in jd_skill_set}
    jd_confidence = {s: v for s, v in e3.skill_confidence.items()
                     if s.lower() in jd_skill_set}

    avg_evidence = (
        float(np.mean(list(jd_evidence.values()))) if jd_evidence
        else (float(np.mean(list(e3.evidence_scores.values()))) if e3.evidence_scores else 0.0)
    )
    avg_confidence = (
        float(np.mean(list(jd_confidence.values()))) if jd_confidence
        else (float(np.mean(list(e3.skill_confidence.values()))) if e3.skill_confidence else 0.0)
    )

    progression_score = progression_to_score(e4.learning_progression)
    readability       = _readability_score(e1.parsed_resume.raw_text)
    ats_compat        = e5.ats_score / 100.0
    edu_score         = _education_score(e1)

    component_scores = {
        "semantic_match":       e2.semantic_score,
        "education_score":      edu_score,
        "experience_quality":   e4.experience_quality_score,
        "project_complexity":   e4.avg_project_complexity_score,
        "evidence_score":       avg_evidence,
        "ats_compatibility":    ats_compat,
        "readability":          readability,
        "learning_progression": progression_score,
    }

    # ── Final score: only active dimensions contribute ────────────────────────
    final = sum(
        component_scores[dim] * w
        for dim, w in weights.items()
        if w > 0
    )

    return ScoreBreakdown(
        semantic_match=round(e2.semantic_score, 4),
        education_score=round(edu_score, 4),
        evidence_score=round(avg_evidence, 4),
        experience_quality=round(e4.experience_quality_score, 4),
        project_complexity=round(e4.avg_project_complexity_score, 4),
        skill_confidence=round(avg_confidence, 4),
        ats_compatibility=round(ats_compat, 4),
        readability=round(readability, 4),
        learning_progression=round(progression_score, 4),
        active_weights=weights,
        final_score=round(final * 100, 2),
    )
