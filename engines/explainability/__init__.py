from core.models import (AnalysisResult, Engine1Output, Engine2Output,
                         Engine3Output, Engine4Output, Engine5Output,
                         Engine6Output)
from engines.recruiter_decision.hiring_recommender import HiringRecommender

from .score_composer import compose
from .shap_explainer import SHAPExplainer
from .suggestion_generator import generate_ai_content


class ExplainabilityEngine:
    def __init__(self):
        self._recommender = HiringRecommender()
        self._shap = SHAPExplainer(self._recommender)

    def run(
        self,
        e1: Engine1Output,
        e2: Engine2Output,
        e3: Engine3Output,
        e4: Engine4Output,
        e5: Engine5Output,
    ) -> Engine6Output:
        score_breakdown = compose(e1, e2, e3, e4, e5)
        shap_values = self._shap.explain(e1, e2, e3, e4)

        partial_result = AnalysisResult(
            resume_intelligence=e1,
            semantic_intelligence=e2,
            candidate_verification=e3,
            candidate_quality=e4,
            recruiter_decision=e5,
            explainability=Engine6Output(
                score_breakdown=score_breakdown,
                shap_values=shap_values,
            ),
        )

        ai = generate_ai_content(partial_result)

        return Engine6Output(
            score_breakdown=score_breakdown,
            shap_values=shap_values,
            improvement_suggestions=ai.get("suggestions", []),
            recruiter_summary=ai.get("recruiter_summary", ""),
            candidate_strengths=ai.get("strengths", []),
            candidate_weaknesses=ai.get("weaknesses", []),
            interview_questions=ai.get("interview_questions", []),
            ai_review=ai.get("ai_review", ""),
        )
