from core.models import Engine1Output, MissingSkill, SkillImportance

from .semantic_matcher import SemanticMatcher


def find_missing_skills(
    e1: Engine1Output,
    matched_skills: dict[str, str],
    matcher: SemanticMatcher,
) -> list[MissingSkill]:
    """
    For each JD skill not covered by a semantic match, compute:
    - current_match_pct: best cosine similarity to any resume skill (as %)
    - estimated_match_pct: if candidate learned this skill (set to 91%)
    """
    resume_skills = e1.extracted_skills.all_skills
    required = set(e1.parsed_jd.required_skills)
    preferred = set(e1.parsed_jd.preferred_skills)
    all_jd_skills = required | preferred

    missing: list[MissingSkill] = []

    for jd_skill in all_jd_skills:
        if jd_skill in matched_skills:
            continue  # already matched

        # Find best current match (semantic)
        best_score = 0.0
        if resume_skills:
            j_vec = matcher.embed_skill(jd_skill)
            for r_skill in resume_skills:
                r_vec = matcher.embed_skill(r_skill)
                sim = matcher.cosine(j_vec, r_vec)
                if sim > best_score:
                    best_score = sim

        current_pct = round(best_score * 100, 1)
        estimated_pct = round(min(current_pct + (91 - current_pct) * 0.7, 91.0), 1)

        missing.append(MissingSkill(
            skill=jd_skill,
            importance=SkillImportance.REQUIRED if jd_skill in required else SkillImportance.PREFERRED,
            current_match_pct=current_pct,
            estimated_match_pct=estimated_pct,
        ))

    # Sort: required first, then by current match ascending (biggest gaps first)
    missing.sort(key=lambda m: (m.importance != SkillImportance.REQUIRED, m.current_match_pct))
    return missing
