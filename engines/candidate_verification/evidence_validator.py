import re

from core.models import Engine1Output
from knowledge_graph.resume_graph import ResumeGraph

# Section weights for evidence scoring
_SECTION_WEIGHTS = {
    "experience":     1.0,
    "projects":       0.9,
    "certifications": 0.7,
    "achievements":   0.5,
    "education":      0.3,
    "skills":         0.1,  # listing alone is weak evidence
}

# Normalise against experience+projects so appearing in both high-value
# sections scores ~1.0 before the graph boost, rather than ~0.54.
_EVIDENCE_NORM = _SECTION_WEIGHTS["experience"] + _SECTION_WEIGHTS["projects"]  # 1.9


def _search_section(skill: str, section_text: str) -> bool:
    return bool(re.search(re.escape(skill), section_text, re.I))


def compute_evidence_scores(
    e1: Engine1Output, graph: ResumeGraph
) -> dict[str, float]:
    """
    For each extracted skill, compute an evidence score 0–1 based on how many
    high-value sections mention it and how often it appears in the graph.
    """
    sections = e1.parsed_resume.sections
    all_skills = e1.extracted_skills.all_skills
    scores: dict[str, float] = {}

    for skill in all_skills:
        raw_score = 0.0

        for section_name, weight in _SECTION_WEIGHTS.items():
            section_text = sections.get(section_name, "")
            if _search_section(skill, section_text):
                raw_score += weight

        # Normalise: appearing in both experience + projects saturates the base.
        base = min(raw_score / _EVIDENCE_NORM, 1.0)

        # Graph boost fills the remaining gap proportionally.
        proj_count = graph.skill_project_count(skill)
        exp_count = graph.skill_experience_count(skill)
        graph_boost = min((proj_count * 0.15 + exp_count * 0.20), 0.40)

        evidence = min(base + graph_boost * (1.0 - base), 1.0)
        scores[skill] = round(evidence, 4)

    return scores
