from core.models import Engine1Output, Engine2Output

from .missing_skills import find_missing_skills
from .role_matcher import compute_role_fit
from .semantic_matcher import SemanticMatcher


class SemanticIntelligenceEngine:
    def __init__(self):
        self.matcher = SemanticMatcher()

    def run(self, e1: Engine1Output) -> Engine2Output:
        resume_text = e1.parsed_resume.raw_text
        jd_text = " ".join(e1.parsed_jd.responsibilities + e1.parsed_jd.required_skills)

        text_sim = self.matcher.score_resume_vs_jd(resume_text, jd_text)

        resume_skills = e1.extracted_skills.all_skills
        jd_skills = e1.parsed_jd.required_skills + e1.parsed_jd.preferred_skills
        matched = self.matcher.match_skills(resume_skills, jd_skills)

        # Skill coverage: required skills carry more weight than preferred.
        # Cap denominators so a very long JD skill list doesn't unfairly
        # collapse coverage (a candidate matching 6 of 8 core skills should
        # not be penalised the same as matching 6 of 24).
        required = set(e1.parsed_jd.required_skills)
        req_matched = sum(1 for s in matched if s in required)
        req_denom = min(max(len(required), 1), 8)
        req_coverage = min(req_matched / req_denom, 1.0)
        all_denom = min(max(len(jd_skills), 1), 15)
        all_coverage = min(len(matched) / all_denom, 1.0)
        if required:
            skill_coverage = 0.7 * req_coverage + 0.3 * all_coverage
        else:
            skill_coverage = all_coverage

        # Blend skill coverage (primary signal) with text similarity (secondary)
        semantic_score = min(0.65 * skill_coverage + 0.35 * text_sim, 1.0)

        missing = find_missing_skills(e1, matched, self.matcher)
        role_fit = compute_role_fit(resume_skills, self.matcher)

        return Engine2Output(
            semantic_score=round(semantic_score, 4),
            matched_skills=matched,
            missing_skills=missing,
            role_fit_scores=role_fit,
        )
