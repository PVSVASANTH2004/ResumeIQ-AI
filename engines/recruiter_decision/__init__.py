import numpy as np

from core.config import ScoreWeights
from core.models import (Engine1Output, Engine2Output, Engine3Output,
                         Engine4Output, Engine5Output, HiringRecommendation)

from .candidate_ranker import CandidateRanker


# Dynamic score thresholds — calibrated to the 0–100 final score range.
_THRESHOLDS = {
    HiringRecommendation.STRONG_HIRE: 80,
    HiringRecommendation.INTERVIEW:   65,
    HiringRecommendation.MAYBE:       50,
}


def _score_to_recommendation(score: float) -> HiringRecommendation:
    if score >= _THRESHOLDS[HiringRecommendation.STRONG_HIRE]:
        return HiringRecommendation.STRONG_HIRE
    if score >= _THRESHOLDS[HiringRecommendation.INTERVIEW]:
        return HiringRecommendation.INTERVIEW
    if score >= _THRESHOLDS[HiringRecommendation.MAYBE]:
        return HiringRecommendation.MAYBE
    return HiringRecommendation.REJECT


def _interview_probability(score: float) -> float:
    """
    Piecewise-linear probability of receiving an interview call.
      Strong Hire (≥80) →  80–98%
      Interview   (65–79) → 60–80%
      Maybe       (50–64) → 28–60%
      Soft reject (25–49) →  3–28%
      Hard reject (<25)   →  0–3%
    """
    if score >= 80:
        return round(min(0.80 + (score - 80) * 0.013, 0.98), 4)
    if score >= 65:
        return round(0.60 + (score - 65) / 15 * 0.20, 4)
    if score >= 50:
        return round(0.28 + (score - 50) / 15 * 0.32, 4)
    if score >= 25:
        return round(0.025 + (score - 25) / 25 * 0.255, 4)
    return round(max(0.0, score * 0.001), 4)


class RecruiterDecisionEngine:
    def __init__(self):
        self.ranker = CandidateRanker()

    def run(
        self,
        e1: Engine1Output,
        e2: Engine2Output,
        e3: Engine3Output,
        e4: Engine4Output,
    ) -> Engine5Output:
        from engines.resume_intelligence.ats_checker import ATSChecker
        ats_score = ATSChecker().score(e1.ats_issues)
        ats_compat = ats_score / 100.0

        # Compute evidence and confidence averages (mirrors score_composer logic)
        avg_evidence = float(np.mean(list(e3.evidence_scores.values()))) \
            if e3.evidence_scores else 0.0

        from engines.candidate_quality.learning_progression import progression_to_score
        from engines.explainability.score_composer import _readability_score, _education_score
        from engines.explainability.jd_intent_detector import detect_jd_intent

        readability = _readability_score(e1.parsed_resume.raw_text)
        progression = progression_to_score(e4.learning_progression)
        edu_score   = _education_score(e1)
        weights     = detect_jd_intent(e1.parsed_jd.raw_text, e1.parsed_jd)

        component_scores = {
            "semantic_match":       e2.semantic_score,
            "education_score":      edu_score,
            "experience_quality":   e4.experience_quality_score,
            "project_complexity":   e4.avg_project_complexity_score,
            "evidence_score":       avg_evidence,
            "ats_compatibility":    ats_compat,
            "readability":          readability,
            "learning_progression": progression,
        }
        prelim_score = sum(
            component_scores[dim] * w
            for dim, w in weights.items()
            if w > 0
        ) * 100

        recommendation = _score_to_recommendation(prelim_score)
        interview_prob = _interview_probability(prelim_score)
        shortlist_prob = _interview_probability(max(prelim_score - 12, 0))

        return Engine5Output(
            ats_score=ats_score,
            hiring_recommendation=recommendation,
            interview_probability=interview_prob,
            shortlist_probability=round(shortlist_prob, 4),
        )

    def rank_bulk(
        self,
        results: list[tuple[str, "AnalysisResult"]],  # noqa: F821
    ) -> list[tuple[str, "AnalysisResult"]]:
        feature_vectors = [
            build_feature_vector(
                r.resume_intelligence,
                r.semantic_intelligence,
                r.candidate_verification,
                r.candidate_quality,
            )
            for _, r in results
        ]
        ranked_idx = self.ranker.rank(feature_vectors)
        return [(results[i][0], results[i][1]) for i in ranked_idx]
