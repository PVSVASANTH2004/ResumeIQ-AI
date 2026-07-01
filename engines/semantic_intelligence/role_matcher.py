from __future__ import annotations

from .semantic_matcher import SemanticMatcher

# Canonical skill sets for each role archetype.
# Kept at ~10 skills so the coverage ratio is meaningful.
ROLE_ARCHETYPES: dict[str, list[str]] = {
    "AI Engineer": [
        "Python", "LangChain", "HuggingFace", "PyTorch", "FastAPI",
        "RAG", "deep learning", "NLP", "embeddings", "LLM",
    ],
    "ML Engineer": [
        "Python", "PyTorch", "TensorFlow", "scikit-learn",
        "deep learning", "NLP", "Keras", "Pandas", "NumPy", "HuggingFace",
    ],
    "Backend Engineer": [
        "Python", "Java", "Node.js", "REST API", "SQL",
        "PostgreSQL", "Docker", "FastAPI", "Django", "Redis",
    ],
    "Frontend Engineer": [
        "React", "JavaScript", "TypeScript", "CSS", "HTML",
        "Angular", "Vue.js", "Redux", "Next.js", "Tailwind",
    ],
    "Full Stack Engineer": [
        "React", "Node.js", "REST API", "SQL", "JavaScript",
        "Python", "MongoDB", "Docker", "PostgreSQL", "Git",
    ],
    "Data Scientist": [
        "Python", "Pandas", "NumPy", "SQL", "scikit-learn",
        "statistics", "R", "Jupyter", "matplotlib", "data analysis",
    ],
}


def compute_role_fit(
    resume_skills: list[str],
    matcher: SemanticMatcher,
) -> dict[str, float]:
    """
    For each role archetype, compute what fraction of its canonical skills
    are semantically covered by the resume.  Returns scores 0–1.
    """
    role_scores: dict[str, float] = {}
    for role_name, archetype_skills in ROLE_ARCHETYPES.items():
        matched = matcher.match_skills(resume_skills, archetype_skills, threshold=0.72)
        coverage = len(matched) / max(len(archetype_skills), 1)
        role_scores[role_name] = round(coverage, 4)
    return role_scores
