import re
from collections import Counter

from core.models import ParsedResume


_STOP_WORDS = {
    "and", "or", "the", "a", "an", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "did", "will", "would", "could",
    "should", "may", "might", "that", "this", "from", "as", "it",
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#\.\-]{1,}\b", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS]


def detect_keyword_stuffing(resume: ParsedResume) -> tuple[float, bool]:
    """
    Returns (risk_score: float 0-1, is_stuffed: bool).

    Heuristics:
    1. Top-5 skill term repetition rate across the full resume text.
    2. Skills section density vs total text length.
    3. Ratio of skill mentions in skills section vs experience/projects.
    """
    full_text = resume.raw_text
    skills_text = resume.sections.get("skills", "")
    experience_text = resume.sections.get("experience", "")
    projects_text = resume.sections.get("projects", "")

    all_tokens = _tokenize(full_text)
    if not all_tokens:
        return 0.0, False

    freq = Counter(all_tokens)
    total_tokens = len(all_tokens)

    # Heuristic 1: repetition rate of top-10 terms
    top10 = freq.most_common(10)
    top10_count = sum(c for _, c in top10)
    repetition_rate = top10_count / total_tokens

    # Heuristic 2: skills section density (skill tokens / total tokens)
    skills_tokens = _tokenize(skills_text)
    skill_density = len(skills_tokens) / max(total_tokens, 1)

    # Heuristic 3: skill terms appearing more in skills section than in evidence sections
    evidence_tokens = set(_tokenize(experience_text + " " + projects_text))
    skills_only_count = sum(
        1 for t in set(skills_tokens) if t not in evidence_tokens
    )
    skills_only_ratio = skills_only_count / max(len(set(skills_tokens)), 1)

    # Weighted risk score
    risk = (
        0.4 * min(repetition_rate * 3, 1.0)
        + 0.2 * min(skill_density * 5, 1.0)
        + 0.4 * skills_only_ratio
    )
    risk = round(min(risk, 1.0), 4)
    is_stuffed = risk >= 0.70

    return risk, is_stuffed
