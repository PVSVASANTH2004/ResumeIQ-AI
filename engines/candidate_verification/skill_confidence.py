from core.models import Engine1Output
from engines.semantic_intelligence.semantic_matcher import SemanticMatcher


def compute_skill_confidence(
    evidence_scores: dict[str, float],
    e1: Engine1Output,
    matcher: SemanticMatcher,
) -> dict[str, float]:
    """
    Confidence = 0.6 × evidence_score + 0.4 × semantic_relevance_to_jd

    semantic_relevance: how closely the skill matches any JD required skill.
    """
    jd_skills = e1.parsed_jd.required_skills + e1.parsed_jd.preferred_skills
    confidence: dict[str, float] = {}

    for skill, ev_score in evidence_scores.items():
        if jd_skills:
            j_vecs = [matcher.embed_skill(js) for js in jd_skills]
            r_vec = matcher.embed_skill(skill)
            semantic_rel = max(matcher.cosine(r_vec, jv) for jv in j_vecs)
        else:
            semantic_rel = ev_score  # fallback

        conf = 0.6 * ev_score + 0.4 * semantic_rel
        confidence[skill] = round(min(conf, 1.0), 4)

    return confidence
