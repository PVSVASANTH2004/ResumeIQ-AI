import re

from core.models import ExperienceEntry

# Engineering action verbs — tiered by how much they signal ownership/impact
_SENIOR_VERBS = re.compile(
    r"\b(designed|architected|led|managed|owned|scaled|spearheaded|"
    r"pioneered|established|launched|shipped|deployed)\b", re.I
)
_STRONG_VERBS = re.compile(
    r"\b(built|developed|implemented|created|engineered|optimized|"
    r"automated|refactored|migrated|integrated|released)\b", re.I
)
_JUNIOR_VERBS = re.compile(
    r"\b(configured|tested|debugged|fixed|resolved|contributed|"
    r"coordinated|collaborated|wrote|added|updated)\b", re.I
)

# Quantified / business impact patterns (added on top of the verb tier base)
_IMPACT_PATTERNS = [
    (r"\d+%\s*(increase|decrease|improvement|reduction|growth|faster)", 0.20),
    (r"\d+x\s*(faster|improvement|growth|increase)",                     0.20),
    (r"\d+[km]\+?\s*(users|customers|requests|transactions|records)",    0.15),
    (r"\$\d+[km]?\s*(saved|generated|revenue|cost)",                     0.15),
    (r"\b(team of \d+|cross.functional|stakeholder)\b",                  0.10),
]

# Technical specificity — naming real tools lifts the score
_TECH_RE = re.compile(
    r"\b(api|rest|backend|frontend|database|sql|authentication|auth|"
    r"cloud|docker|pipeline|microservice|endpoint|integration|jwt|"
    r"firebase|aws|azure|react|python|fastapi|node|express|mongodb|"
    r"postgres|redis|kafka|ci.cd|webhook|deployment|serverless)\b",
    re.I,
)

_WEAK_PATTERNS = [
    r"\bworked on\b",
    r"\bhelped with\b",
    r"\bassisted\b",
    r"\bresponsible for\b",
    r"\binvolved in\b",
]


def _score_bullet(bullet: str) -> float:
    text = bullet.lower()
    words = bullet.split()

    # ── Tier base ─────────────────────────────────────────────────────────────
    if _SENIOR_VERBS.search(text):
        score = 0.50          # leadership / ownership language
    elif _STRONG_VERBS.search(text):
        score = 0.40          # solid engineering contribution
    elif _JUNIOR_VERBS.search(text):
        score = 0.28          # valid junior/intern work
    elif len(words) > 4:
        score = 0.12          # some content, no recognisable verb
    else:
        score = 0.0

    # ── Technical specificity bonus (up to +0.15) ─────────────────────────────
    tech_hits = len(_TECH_RE.findall(text))
    score += min(tech_hits * 0.05, 0.15)

    # ── Quantified impact premium ─────────────────────────────────────────────
    for pattern, weight in _IMPACT_PATTERNS:
        if re.search(pattern, text, re.I):
            score += weight

    # ── Penalty for passive / vague language ──────────────────────────────────
    weak_count = sum(1 for p in _WEAK_PATTERNS if re.search(p, text, re.I))
    score -= weak_count * 0.10

    return max(0.0, min(score, 1.0))


def compute_experience_quality(experience: list[ExperienceEntry]) -> float:
    """
    Aggregate experience quality score 0–1.
    Accounts for: bullet impact, tenure, number of roles.
    """
    if not experience:
        return 0.0

    role_scores: list[float] = []
    for exp in experience:
        if not exp.bullets:
            role_scores.append(0.1)
            continue
        bullet_scores = [_score_bullet(b) for b in exp.bullets]
        avg_bullet = sum(bullet_scores) / len(bullet_scores)

        # Tenure bonus (capped at 24 months = 1.0)
        tenure_bonus = min(exp.duration_months / 24.0, 1.0) * 0.15

        role_scores.append(min(avg_bullet + tenure_bonus, 1.0))

    return round(sum(role_scores) / len(role_scores), 4)
