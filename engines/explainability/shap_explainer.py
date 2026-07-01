from __future__ import annotations

import numpy as np

from core.models import (Engine1Output, Engine2Output, Engine3Output,
                         Engine4Output, Engine5Output)
from engines.recruiter_decision.hiring_recommender import (
    HiringRecommender, build_feature_vector)

_FEATURE_NAMES = [
    "Semantic Match",
    "Evidence Score",
    "Experience Quality",
    "Project Complexity",
    "Skill Confidence",
    "Hallucination Rate",
    "Keyword Stuffing",
    "Skill Breadth",
    "Experience Count",
    "Project Count",
    "Missing Required Skills",
    "Total Tenure",
]

# Contribution sign (positive = boosts score, negative = hurts score)
_FEATURE_SIGN = [1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1]


class SHAPExplainer:
    """
    Uses shap.TreeExplainer when the XGBoost model is trained.
    Falls back to a linear approximation using ScoreWeights when not.
    """

    def __init__(self, recommender: HiringRecommender):
        self._recommender = recommender
        self._explainer = None
        self._try_init_shap()

    def _try_init_shap(self):
        if self._recommender._model is None:
            return
        try:
            import shap
            self._explainer = shap.TreeExplainer(self._recommender._model)
        except Exception:
            pass

    def explain(
        self,
        e1: Engine1Output,
        e2: Engine2Output,
        e3: Engine3Output,
        e4: Engine4Output,
    ) -> dict[str, float]:
        """Return {feature_name: shap_value} dict (positive = helps, negative = hurts)."""
        fv = build_feature_vector(e1, e2, e3, e4)

        if self._explainer is not None:
            try:
                import shap
                shap_vals = self._explainer.shap_values(fv.reshape(1, -1))
                # For multiclass, take the "Strong Hire" class values (index 3)
                if isinstance(shap_vals, list):
                    vals = shap_vals[3][0]
                else:
                    vals = shap_vals[0]
                return {name: round(float(v), 4) for name, v in zip(_FEATURE_NAMES, vals)}
            except Exception:
                pass

        # Linear fallback: feature value × weight × sign
        weights = np.array([
            0.30, 0.20, 0.15, 0.10, 0.10, 0.10, 0.05,
            0.03, 0.03, 0.03, 0.03, 0.01
        ], dtype=np.float32)

        contributions = fv * weights * np.array(_FEATURE_SIGN, dtype=np.float32)
        return {name: round(float(c), 4) for name, c in zip(_FEATURE_NAMES, contributions)}
