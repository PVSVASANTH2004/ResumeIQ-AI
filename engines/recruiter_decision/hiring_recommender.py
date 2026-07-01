from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from core.config import ModelPaths
from core.models import (Engine1Output, Engine2Output, Engine3Output,
                         Engine4Output, HiringRecommendation)


_LABELS = [
    HiringRecommendation.REJECT,
    HiringRecommendation.MAYBE,
    HiringRecommendation.INTERVIEW,
    HiringRecommendation.STRONG_HIRE,
]


def build_feature_vector(
    e1: Engine1Output,
    e2: Engine2Output,
    e3: Engine3Output,
    e4: Engine4Output,
) -> np.ndarray:
    """
    12-feature vector consumed by XGBoost and LightGBM.
    Keep this function as the single source of truth for feature construction.
    """
    avg_evidence = np.mean(list(e3.evidence_scores.values())) if e3.evidence_scores else 0.0
    avg_confidence = np.mean(list(e3.skill_confidence.values())) if e3.skill_confidence else 0.0
    hallucination_rate = len(e3.hallucinated_skills) / max(len(e1.extracted_skills.all_skills), 1)
    missing_required = sum(
        1 for ms in e2.missing_skills if ms.importance.value == "required"
    )

    return np.array([
        e2.semantic_score,                       # 0: semantic match
        avg_evidence,                            # 1: evidence score
        e4.experience_quality_score,             # 2: experience quality
        e4.avg_project_complexity_score,         # 3: project complexity
        avg_confidence,                          # 4: skill confidence
        hallucination_rate,                      # 5: hallucination rate (negative)
        e1.stuffing_risk,                        # 6: keyword stuffing (negative)
        len(e1.extracted_skills.all_skills) / 30.0,  # 7: skill breadth
        len(e1.parsed_resume.experience) / 5.0,      # 8: experience count
        len(e1.parsed_resume.projects) / 5.0,        # 9: project count
        missing_required / 5.0,                  # 10: missing required skills (negative)
        min(sum(e.duration_months for e in e1.parsed_resume.experience) / 48.0, 1.0),  # 11: total tenure
    ], dtype=np.float32)


class HiringRecommender:
    def __init__(self):
        self._model = None
        self._load_model()

    def _load_model(self):
        path = Path(ModelPaths.HIRING_RECOMMENDER)
        if path.exists():
            with open(path, "rb") as f:
                self._model = pickle.load(f)["model"]

    def predict(self, feature_vector: np.ndarray) -> tuple[HiringRecommendation, np.ndarray]:
        """Returns (recommendation, class_probabilities)."""
        if self._model is not None:
            proba = self._model.predict_proba(feature_vector.reshape(1, -1))[0]
            label_idx = int(np.argmax(proba))
            return _LABELS[label_idx], proba

        # Heuristic fallback with soft probabilities
        weighted = (
            feature_vector[0] * 0.30  # semantic
            + feature_vector[1] * 0.20  # evidence
            + feature_vector[2] * 0.15  # experience quality
            + feature_vector[3] * 0.10  # project complexity
            + feature_vector[4] * 0.10  # skill confidence
            - feature_vector[5] * 0.10  # hallucination rate
            - feature_vector[6] * 0.05  # keyword stuffing
        )
        weighted = float(np.clip(weighted, 0.0, 1.0))

        # Soft class probabilities via triangular membership functions
        # [Reject, Maybe, Interview, StrongHire] centred at 0.20, 0.42, 0.62, 0.82
        centres = np.array([0.20, 0.42, 0.62, 0.82], dtype=np.float32)
        raw = np.exp(-((weighted - centres) ** 2) / (2 * 0.12 ** 2))
        proba = (raw / raw.sum()).astype(np.float32)

        label_idx = int(np.argmax(proba))
        return _LABELS[label_idx], proba
