import re

from core.models import ParsedJD

_DEGREE_RE = re.compile(
    r"\b(bachelor|master|phd|ph\.d|doctorate|b\.tech|b\.e\.|m\.tech|mba|"
    r"undergraduate|postgraduate|graduate|degree|diploma|gpa)\b", re.I
)
_RESEARCH_RE = re.compile(
    r"\b(research|publication|paper|arxiv|dissertation|thesis|conference|journal|ieee|acm)\b", re.I
)
_EXPERIENCE_RE = re.compile(
    r"\d+\+?\s*years?\s*of\s*(professional\s*)?experience|"
    r"work\s*experience|industry\s*experience", re.I
)


def detect_jd_intent(jd_text: str, parsed_jd: ParsedJD) -> dict[str, float]:
    """
    Analyse JD content and return scoring dimension weights that sum to 1.0.

    Intent classes
    --------------
    education-only  → lots of degree/GPA keywords, few tech skills
    research        → research/publication keywords prominent
    mixed           → education requirements + tech skills both present
    standard        → typical software engineering JD (default)
    """
    edu_hits      = len(_DEGREE_RE.findall(jd_text))
    research_hits = len(_RESEARCH_RE.findall(jd_text))
    skills_count  = len(parsed_jd.required_skills) + len(parsed_jd.preferred_skills)

    edu_strength      = min(edu_hits / 4.0, 1.0)
    skills_strength   = min(skills_count / 5.0, 1.0)
    research_strength = min(research_hits / 4.0, 1.0)

    # Education-only JD: degree/GPA keywords dominate, few/no tech skills
    if edu_strength >= 0.5 and skills_strength < 0.3:
        return {
            "semantic_match":        0.05,
            "education_score":       0.80,
            "experience_quality":    0.00,
            "project_complexity":    0.00,
            "evidence_score":        0.00,
            "ats_compatibility":     0.10,
            "readability":           0.05,
            "learning_progression":  0.00,
        }

    # Research JD: research/publication keywords prominent
    if research_strength >= 0.4:
        return {
            "semantic_match":        0.20,
            "education_score":       0.10,
            "experience_quality":    0.10,
            "project_complexity":    0.30,
            "evidence_score":        0.15,
            "ats_compatibility":     0.05,
            "readability":           0.05,
            "learning_progression":  0.05,
        }

    # Mixed JD: education requirements alongside tech skills
    if edu_strength >= 0.3 and skills_strength >= 0.3:
        return {
            "semantic_match":        0.28,
            "education_score":       0.15,
            "experience_quality":    0.17,
            "project_complexity":    0.12,
            "evidence_score":        0.10,
            "ats_compatibility":     0.08,
            "readability":           0.05,
            "learning_progression":  0.05,
        }

    # Default: standard software engineering JD
    return {
        "semantic_match":        0.35,
        "education_score":       0.00,
        "experience_quality":    0.20,
        "project_complexity":    0.15,
        "evidence_score":        0.10,
        "ats_compatibility":     0.10,
        "readability":           0.05,
        "learning_progression":  0.05,
    }
