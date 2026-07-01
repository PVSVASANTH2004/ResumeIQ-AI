from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

from core.config import ModelPaths, Thresholds


class HallucinationDetector:
    """
    Isolation Forest trained on the feature vector:
    [evidence_score, skill_confidence, section_mention_count, stuffing_risk]

    An anomaly = claimed skill with very low evidence + high stuffing risk.
    Falls back to a threshold heuristic if no trained model exists.
    """

    def __init__(self):
        self._model: IsolationForest | None = None
        self._load_model()

    def _load_model(self):
        path = Path(ModelPaths.HALLUCINATION_DETECTOR)
        if path.exists():
            with open(path, "rb") as f:
                self._model = pickle.load(f)

    def _feature_vector(
        self,
        skill: str,
        evidence_scores: dict[str, float],
        confidence: dict[str, float],
        stuffing_risk: float,
    ) -> list[float]:
        ev = evidence_scores.get(skill, 0.0)
        conf = confidence.get(skill, 0.0)
        gap = conf - ev  # positive gap → claimed high confidence, low evidence
        return [ev, conf, gap, stuffing_risk]

    def detect(
        self,
        skills: list[str],
        evidence_scores: dict[str, float],
        confidence: dict[str, float],
        stuffing_risk: float,
    ) -> list[str]:
        """Return list of skill names flagged as potentially hallucinated."""
        if not skills:
            return []

        flagged: list[str] = []

        if self._model is not None:
            X = np.array([
                self._feature_vector(s, evidence_scores, confidence, stuffing_risk)
                for s in skills
            ])
            preds = self._model.predict(X)  # -1 = anomaly, 1 = normal
            flagged = [s for s, p in zip(skills, preds) if p == -1]
        else:
            # Heuristic fallback: low evidence + low confidence
            for skill in skills:
                ev = evidence_scores.get(skill, 0.0)
                conf = confidence.get(skill, 0.0)
                if ev < Thresholds.EVIDENCE_REQUIRED and conf < Thresholds.SKILL_CONFIDENCE_LOW:
                    flagged.append(skill)

        return flagged
